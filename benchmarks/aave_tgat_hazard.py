"""Chronological borrower-level prediction around an Aave liquidation event.

The event study establishes graph-local risk, not causal contagion. This benchmark asks whether
TGAT can rank *which* exposed borrower liquidates next beyond explicit path
membership, current portfolio degree and prior-liquidation history.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine
from ingestion.aave import load_aave_events
from models import PCSRTemporalSampler, ReturnForecastModel
from benchmarks.aave_liquidation_contagion import build_candidate_panel


RELATION = {
    "supply": 1, "withdraw": 2, "borrow": 3, "repay": 4,
}


def build_graph(events):
    user_codes, users = pd.factorize(events["user"], sort=False)
    asset_codes, assets = pd.factorize(events["asset"], sort=False)
    user_codes = user_codes.astype(np.uint32)
    asset_nodes = (asset_codes + len(users)).astype(np.uint32)
    relation = np.zeros(len(events), dtype=np.uint16)
    for typ, code in RELATION.items():
        relation[events["event_type"].eq(typ).to_numpy()] = code
    liquidation = events["event_type"].eq("liquidationCall").to_numpy()
    collateral = events["asset_role"].eq("collateral").to_numpy()
    relation[liquidation & collateral] = 5
    relation[liquidation & ~collateral] = 6
    if np.any(relation == 0):
        unknown = sorted(events.loc[relation == 0, "event_type"].unique())
        raise ValueError(f"unmapped Aave event types: {unknown}")

    times = events["timestamp"].to_numpy(dtype=np.uint32)
    src = np.column_stack([user_codes, asset_nodes]).reshape(-1)
    dst = np.column_stack([asset_nodes, user_codes]).reshape(-1)
    ts = np.repeat(times, 2)
    rel = np.column_stack([relation, relation + 6]).reshape(-1)
    vertices = len(users) + len(assets)
    capacity = int(len(src) * 2.2)
    graph = graph_engine.PCSRGraph(vertices, capacity,
                                  capacity * 8 * 4 + (1 << 26))
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32), ts, rel)
    sampler = PCSRTemporalSampler(graph)
    if not sampler.validate():
        raise RuntimeError("Aave graph is not chronological")
    user_map = {str(user): index for index, user in enumerate(users)}
    return graph, sampler, user_map, len(assets), len(src)


def features(frame):
    return np.column_stack([
        np.log1p(frame["degree"].to_numpy()),
        np.log1p(frame["prior_liquidations"].to_numpy()),
        np.log1p(frame["exact_pool_size"].to_numpy()),
        frame["group"].eq("exact").to_numpy(dtype=float),
        frame["group"].eq("one_leg").to_numpy(dtype=float),
    ])


def train_tgat(tag, sampler, frame, user_ids, target, layers, epochs,
               max_train, seed=0):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(target)
    negative = np.flatnonzero(~target)
    max_negative = min(len(negative), max(len(positive) * 10, 1))
    negative = rng.choice(negative, max_negative, replace=False)
    take = np.concatenate([positive, negative])
    if len(take) > max_train:
        # Preserve every positive if possible, then fill with controls.
        budget = max(max_train - len(positive), 0)
        take = np.concatenate([positive[:max_train], rng.choice(negative, budget,
                                                                 replace=False)])
    take = rng.permutation(take)
    nodes = user_ids[take]
    times = frame["seed_time"].to_numpy(dtype=np.int64)[take]
    labels = np.where(target[take], 0.5, -0.5).astype(np.float32)

    torch.manual_seed(seed)
    model = ReturnForecastModel(
        sampler.num_vertices, sampler, task="classification",
        use_recency_features=True, node_dim=32, time_dim=32,
        num_layers=layers, num_neighbors=10, num_heads=2, dropout=.1,
        num_relations=13, relation_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(nodes)); losses = []
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            if len(batch) < 16:
                continue
            optimizer.zero_grad()
            loss = model.loss(nodes[batch], times[batch], labels[batch])
            loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        print(f"  {tag} epoch {epoch + 1}/{epochs}: loss {np.mean(losses):.4f}",
              flush=True)
    return model


def predict(model, ids, frame):
    times = frame["seed_time"].to_numpy(dtype=np.int64)
    result = []
    for start in range(0, len(frame), 512):
        result.append(model.predict(ids[start:start + 512], times[start:start + 512]))
    return np.concatenate(result)


def report(name, target, score):
    auc = roc_auc_score(target, score)
    ap = average_precision_score(target, score)
    cutoff = np.quantile(score, .90)
    recall = float(target[score >= cutoff].sum() / max(target.sum(), 1))
    lift = float(target[score >= cutoff].mean() / max(target.mean(), 1e-12))
    print(f"{name:<24} AUC {auc:.3f} | AP {ap:.3f} | "
          f"top-decile recall {recall:.1%} / lift {lift:.2f}x")
    return auc, ap


def clustered_bootstrap(target, baseline, challenger, times, draws=500, seed=0):
    """Date-block bootstrap for paired AUC/AP improvements."""
    dates = pd.to_datetime(times, unit="s", utc=True).date
    unique = np.unique(dates)
    rows = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    auc_diff, ap_diff = [], []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([rows[date] for date in sampled])
        y = target[index]
        if y.min() == y.max():
            continue
        auc_diff.append(roc_auc_score(y, challenger[index])
                        - roc_auc_score(y, baseline[index]))
        ap_diff.append(average_precision_score(y, challenger[index])
                       - average_precision_score(y, baseline[index]))
    return (np.quantile(auc_diff, [.025, .975]),
            np.quantile(ap_diff, [.025, .975]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "data/amm_events/AaveEventData"))
    parser.add_argument("--panel", default=os.path.join(
        ROOT, "data/amm_events/prepared/candidates_seed0.parquet"))
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    events, liquidations = load_aave_events(args.data)
    if os.path.exists(args.panel):
        panel = pd.read_parquet(args.panel)
    else:
        panel = build_candidate_panel(events, liquidations)
    panel = panel.sort_values(["seed_block", "group", "user"]).reset_index(drop=True)
    seeds = np.sort(panel["seed_block"].unique())
    train_end, val_end = seeds[int(.60 * len(seeds))], seeds[int(.80 * len(seeds))]
    train = panel[panel["seed_block"] < train_end].copy()
    validation = panel[(panel["seed_block"] >= train_end)
                       & (panel["seed_block"] < val_end)].copy()
    test = panel[panel["seed_block"] >= val_end].copy()
    for frame in (train, validation, test):
        frame["target"] = frame["next_block"] <= frame["seed_block"] + args.horizon
    print(f"Aave h={args.horizon}: train {len(train):,} | validation {len(validation):,} | "
          f"test {len(test):,} | holdout prevalence {test['target'].mean():.2%}")

    baseline = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, learning_rate=.05,
        l2_regularization=1, random_state=args.seed)
    baseline.fit(features(train), train["target"])
    base_val = baseline.predict_proba(features(validation))[:, 1]
    base_test = baseline.predict_proba(features(test))[:, 1]
    report("explicit-path baseline", test["target"].to_numpy(), base_test)

    graph, sampler, user_map, asset_count, edge_count = build_graph(events)
    for frame in (train, validation, test):
        frame["user_id"] = frame["user"].map(user_map)
        if frame["user_id"].isna().any():
            raise RuntimeError("candidate user absent from event graph")
    print(f"graph: {len(user_map):,} users | {asset_count} assets | "
          f"{edge_count:,} directed typed events")

    train_ids = train["user_id"].to_numpy(dtype=np.int64)
    val_ids = validation["user_id"].to_numpy(dtype=np.int64)
    test_ids = test["user_id"].to_numpy(dtype=np.int64)
    one = train_tgat("one-hop", sampler, train, train_ids,
                     train["target"].to_numpy(), 1, args.epochs, args.max_train,
                     seed=args.seed)
    one_val, one_test = predict(one, val_ids, validation), predict(one, test_ids, test)
    report("one-hop TGAT", test["target"].to_numpy(), one_test)

    two = train_tgat("two-hop", sampler, train, train_ids,
                     train["target"].to_numpy(), 2, args.epochs, args.max_train,
                     seed=args.seed)
    two_val, two_test = predict(two, val_ids, validation), predict(two, test_ids, test)
    report("two-hop TGAT", test["target"].to_numpy(), two_test)

    combiner = LogisticRegression(C=1).fit(
        np.column_stack([base_val, two_val]), validation["target"])
    combined = combiner.predict_proba(np.column_stack([base_test, two_test]))[:, 1]
    report("baseline + TGAT", test["target"].to_numpy(), combined)
    print(f"combiner coefficients: explicit path {combiner.coef_[0, 0]:+.3f}, "
          f"TGAT {combiner.coef_[0, 1]:+.3f}")
    auc_ci, ap_ci = clustered_bootstrap(
        test["target"].to_numpy(), base_test, combined,
        test["seed_time"].to_numpy(), seed=args.seed)
    print(f"combined minus baseline, date bootstrap: "
          f"AUC 95% CI [{auc_ci[0]:+.3f}, {auc_ci[1]:+.3f}] | "
          f"AP 95% CI [{ap_ci[0]:+.3f}, {ap_ci[1]:+.3f}]")
    if args.out:
        np.savez(args.out, target=test["target"].to_numpy(), baseline=base_test,
                 one_hop=one_test, two_hop=two_test, combined=combined,
                 seed_time=test["seed_time"].to_numpy(),
                 seed_block=test["seed_block"].to_numpy())


if __name__ == "__main__":
    main()
