"""A temporal self-event graph for pre-earnings tail-risk prediction.

Each company is a node and each completed earnings reaction becomes a typed
self-edge only after that reaction is observable.  Querying the company at its
next announcement therefore exposes an ordered, leakage-safe event history.
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
from models import PCSRTemporalSampler, ReturnForecastModel
from models.earnings_distribution import build_pre_event_features


def timestamp_seconds(frame):
    clock = frame["anntims_act"].astype("string").fillna("12:00:00")
    local = pd.to_datetime(frame["anndats_act"].dt.strftime("%Y-%m-%d") + " " + clock)
    return (local.dt.tz_localize("America/New_York", ambiguous="NaT",
                                nonexistent="shift_forward")
            .dt.tz_convert("UTC").astype("int64").floordiv(10**9).to_numpy())


def build_self_event_graph(frame, randomize_relations=False, seed=0):
    companies = np.sort(frame["permno"].unique())
    company_id = {int(permno): i for i, permno in enumerate(companies)}
    nodes = frame["permno"].map(company_id).to_numpy(dtype=np.uint32)

    # Edge availability is the close of the reaction trading day, never the
    # announcement instant. This prevents a release from describing itself.
    availability = (pd.to_datetime(frame["reaction_date"]).dt.tz_localize(
        "America/New_York") + pd.Timedelta(hours=16, minutes=1))
    times = availability.dt.tz_convert("UTC").astype("int64").floordiv(
        10**9).to_numpy(dtype=np.uint32)
    magnitude = np.digitize(frame["abs_reaction_1d"].to_numpy(), [.05, .10, .20])
    direction = frame["reaction_1d"].ge(0).to_numpy(dtype=np.uint16)
    surprise = frame["eps_surprise_z"].ge(0).to_numpy(dtype=np.uint16)
    relation = (1 + magnitude + 4 * direction + 8 * surprise).astype(np.uint16)
    if randomize_relations:
        # Independent labels destroy outcome content without moving a future
        # realized outcome onto an earlier edge. Permuting observed labels
        # within a company would be a subtle look-ahead counterfactual.
        rng = np.random.default_rng(seed)
        relation = rng.integers(1, 17, size=len(relation), dtype=np.uint16)

    order = np.lexsort((times, nodes))
    capacity = max(int(len(order) * 2.2), len(order) + 1024)
    graph = graph_engine.PCSRGraph(
        len(companies), capacity, capacity * 8 * 4 + (1 << 25))
    graph.insert_edges(nodes[order], nodes[order], times[order], relation[order])
    sampler = PCSRTemporalSampler(graph)
    if not sampler.validate():
        raise RuntimeError("earnings self-event graph is not chronological")
    return graph, sampler, company_id


def train_model(sampler, frame, nodes, target, typed, epochs, max_train, seed):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(target)
    negative = np.flatnonzero(~target)
    negative = rng.choice(negative, min(len(negative), len(positive) * 6), replace=False)
    take = np.concatenate([positive, negative])
    if len(take) > max_train:
        take = rng.choice(take, max_train, replace=False)
    take = rng.permutation(take)
    times = timestamp_seconds(frame)
    labels = np.where(target, .5, -.5).astype(np.float32)

    torch.manual_seed(seed)
    model = ReturnForecastModel(
        sampler.num_vertices, sampler, task="classification",
        use_recency_features=True, node_dim=32, time_dim=32, num_layers=1,
        num_neighbors=12, num_heads=2, dropout=.1,
        num_relations=17 if typed else 0, relation_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5)
    model.train()
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(take), 256):
            batch = take[start:start + 256]
            if len(batch) < 16:
                continue
            optimizer.zero_grad()
            loss = model.loss(nodes[batch], times[batch], labels[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        take = rng.permutation(take)
        print(f"  epoch {epoch + 1}/{epochs}: loss {np.mean(losses):.4f}", flush=True)
    return model


def predict(model, frame, nodes):
    times = timestamp_seconds(frame)
    chunks = [model.predict(nodes[start:start + 512], times[start:start + 512]) + .5
              for start in range(0, len(frame), 512)]
    return np.concatenate(chunks)


def report(name, target, score):
    auc = roc_auc_score(target, score)
    ap = average_precision_score(target, score)
    threshold = np.quantile(score, .90)
    lift = target[score >= threshold].mean() / target.mean()
    print(f"{name:<24} AUC {auc:.3f} | AP {ap:.3f} | top10 lift {lift:.2f}x")
    return auc, ap


def date_bootstrap(frame, baseline, challenger, repetitions=500, seed=0):
    dates = pd.to_datetime(frame["anndats_act"]).dt.normalize().to_numpy()
    unique = np.unique(dates)
    rows = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    target = frame["target"].to_numpy()
    for _ in range(repetitions):
        index = np.concatenate([rows[date] for date in rng.choice(
            unique, len(unique), replace=True)])
        truth = target[index]
        if truth.min() == truth.max():
            continue
        auc.append(roc_auc_score(truth, challenger[index])
                   - roc_auc_score(truth, baseline[index]))
        ap.append(average_precision_score(truth, challenger[index])
                  - average_precision_score(truth, baseline[index]))
    return np.quantile(auc, [.025, .5, .975]), np.quantile(ap, [.025, .5, .975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--tail", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--out", default=os.path.join(
        ROOT, "data/earnings/tgat_tail_predictions.npz"))
    args = parser.parse_args()

    raw = pd.read_parquet(args.data)
    frame, features = build_pre_event_features(raw)
    train = frame[frame["anndats_act"] <= "2021-12-31"].copy()
    validation = frame[(frame["anndats_act"] > "2021-12-31")
                       & (frame["anndats_act"] <= "2022-12-31")].copy()
    test = frame[frame["anndats_act"] > "2022-12-31"].copy()
    for split in (train, validation, test):
        split["target"] = split["abs_reaction_1d"] >= args.tail
    print(f"tail {args.tail:.0%}: train {len(train):,} | val {len(validation):,} | "
          f"test {len(test):,}, prevalence {test['target'].mean():.1%}")

    baseline = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=31, learning_rate=.05,
        l2_regularization=2, random_state=args.seed)
    baseline.fit(train[features], train["target"])
    base_val = baseline.predict_proba(validation[features])[:, 1]
    base_test = baseline.predict_proba(test[features])[:, 1]
    report("scalar temporal", test["target"].to_numpy(), base_test)

    graph, sampler, company_map = build_self_event_graph(frame)
    for split in (train, validation, test):
        split["node_id"] = split["permno"].map(company_map)
    train_nodes = train["node_id"].to_numpy(dtype=np.int64)
    val_nodes = validation["node_id"].to_numpy(dtype=np.int64)
    test_nodes = test["node_id"].to_numpy(dtype=np.int64)
    print(f"self-event graph: {len(company_map):,} companies | {len(frame):,} typed events")

    typed = train_model(sampler, train, train_nodes, train["target"].to_numpy(),
                        True, args.epochs, args.max_train, args.seed)
    typed_val = predict(typed, validation, val_nodes)
    typed_test = predict(typed, test, test_nodes)
    report("typed self-event TGAT", test["target"].to_numpy(), typed_test)

    untyped_test = shuffled_test = np.full(len(test), np.nan)
    if not args.skip_controls:
        untyped = train_model(sampler, train, train_nodes,
                              train["target"].to_numpy(), False, args.epochs,
                              args.max_train, args.seed)
        untyped_test = predict(untyped, test, test_nodes)
        report("untyped self-event TGAT", test["target"].to_numpy(), untyped_test)

        shuffled_graph, shuffled_sampler, shuffled_map = build_self_event_graph(
            frame, randomize_relations=True, seed=args.seed)
        shuffled_train_nodes = train["permno"].map(shuffled_map).to_numpy(dtype=np.int64)
        shuffled_test_nodes = test["permno"].map(shuffled_map).to_numpy(dtype=np.int64)
        shuffled = train_model(
            shuffled_sampler, train, shuffled_train_nodes,
            train["target"].to_numpy(), True, args.epochs, args.max_train, args.seed)
        shuffled_test = predict(shuffled, test, shuffled_test_nodes)
        report("relation-random TGAT", test["target"].to_numpy(), shuffled_test)

    combiner = LogisticRegression(C=1).fit(
        np.column_stack([base_val, typed_val]), validation["target"])
    combined = combiner.predict_proba(np.column_stack([base_test, typed_test]))[:, 1]
    report("scalar + TGAT", test["target"].to_numpy(), combined)
    print(f"combiner weights: scalar {combiner.coef_[0, 0]:+.3f}, "
          f"TGAT {combiner.coef_[0, 1]:+.3f}")
    auc_ci, ap_ci = date_bootstrap(test, base_test, combined, seed=args.seed)
    print(f"combined minus scalar, date bootstrap: AUC "
          f"[{auc_ci[0]:+.4f}, {auc_ci[2]:+.4f}], median {auc_ci[1]:+.4f} | "
          f"AP [{ap_ci[0]:+.4f}, {ap_ci[2]:+.4f}], median {ap_ci[1]:+.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, target=test["target"].to_numpy(), baseline=base_test,
             tgat=typed_test, untyped=untyped_test, shuffled=shuffled_test,
             combined=combined,
             timestamp=timestamp_seconds(test), event_id=test["event_id"].to_numpy())


if __name__ == "__main__":
    main()
