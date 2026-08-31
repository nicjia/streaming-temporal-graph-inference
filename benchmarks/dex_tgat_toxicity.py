"""Does a two-hop event graph add to ordinary DEX toxicity features?

Pool histories alone can predict some next-block markout because price gaps and
flow cluster.  This benchmark asks the harder, project-relevant question: does
the identity and ordered cross-pool history of the actors add anything?

The graph is bipartite and directed in both directions::

    pool A -> actor/searcher -> pool B

At a pool query, one hop sees recent actors; two hops sees where those actors
were immediately beforehand.  The actor-shuffled control preserves every
pool's event times and buy/sell relations while breaking those paths.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine
from ingestion.dex import (ORDER_SLOTS_PER_BLOCK, add_cross_pool_markout,
                           load_chainticks)
from models import PCSRTemporalSampler, ReturnForecastModel
from benchmarks.dex_toxicity import build_block_panel


def build_graph(swaps, shuffle_actors: bool = False, seed: int = 0):
    actors, actor_names = np.unique(swaps["sender"].astype(str), return_inverse=True)
    actor_ids = actor_names.astype(np.uint32)
    if shuffle_actors:
        # Permute actor identities *inside each block*.  The pool, timestamp,
        # direction and activity process are unchanged, and no future event is
        # moved before a query.  Only time-respecting actor paths are broken.
        actor_ids = actor_ids.copy()
        rng = np.random.default_rng(seed)
        blocks = swaps["block"].to_numpy()
        cuts = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1, len(blocks)]
        for left, right in zip(cuts[:-1], cuts[1:]):
            actor_ids[left:right] = rng.permutation(actor_ids[left:right])

    pool_nodes = len(actors) + swaps["pool"].to_numpy(dtype=np.uint32)
    times = swaps["event_time"].to_numpy(dtype=np.uint32)
    direction = swaps["direction"].to_numpy()
    outward_relation = np.where(direction > 0, 1, 2).astype(np.uint16)
    inward_relation = np.where(direction > 0, 3, 4).astype(np.uint16)

    # Interleave the two directions so insertion remains globally temporal.
    src = np.column_stack([actor_ids, pool_nodes]).reshape(-1)
    dst = np.column_stack([pool_nodes, actor_ids]).reshape(-1)
    ts = np.repeat(times, 2)
    relation = np.column_stack([outward_relation, inward_relation]).reshape(-1)
    vertices = len(actors) + 2
    capacity = int(len(src) * 2.2)
    graph = graph_engine.PCSRGraph(
        vertices, capacity, capacity * 8 * 4 + (1 << 24))
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32), ts, relation)
    sampler = PCSRTemporalSampler(graph)
    if not sampler.validate():
        raise RuntimeError("DEX graph adjacency is not chronological")
    return graph, sampler, len(actors), len(src)


def train_tgat(tag, sampler, pool_node, train, threshold, layers,
               epochs, max_train, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    query_time = train["query_time"].to_numpy(dtype=np.int64)
    targets = (train["next_markout"].to_numpy() > threshold).astype(np.float32) - 0.5

    if len(train) > max_train:
        take = np.sort(rng.choice(len(train), max_train, replace=False))
        query_time, targets = query_time[take], targets[take]
    nodes = np.full(len(query_time), pool_node, dtype=np.int64)
    model = ReturnForecastModel(
        sampler.num_vertices, sampler, task="classification",
        use_recency_features=True, node_dim=32, time_dim=32,
        num_layers=layers, num_neighbors=10, num_heads=2, dropout=0.1,
        num_relations=5, relation_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(nodes))
        losses = []
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            if len(batch) < 16:
                continue
            optimizer.zero_grad()
            loss = model.loss(nodes[batch], query_time[batch], targets[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"  {tag} epoch {epoch + 1}/{epochs}: loss {np.mean(losses):.4f}",
              flush=True)
    return model


def predict(model, pool_node, frame):
    nodes = np.full(len(frame), pool_node, dtype=np.int64)
    times = frame["query_time"].to_numpy(dtype=np.int64)
    out = []
    for start in range(0, len(frame), 512):
        # predict returns P(class)-0.5, which is still a monotone score.
        out.append(model.predict(nodes[start:start + 512], times[start:start + 512]))
    return np.concatenate(out)


def metrics(name, target, score):
    auc = roc_auc_score(target, score)
    ap = average_precision_score(target, score)
    print(f"{name:<24} AUC {auc:.3f} | AP {ap:.3f}")
    return auc, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "data/chainticks"))
    parser.add_argument("--pool", choices=["v2", "v3"], default="v2")
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=30_000)
    args = parser.parse_args()

    swaps = add_cross_pool_markout(load_chainticks(args.data), 3)
    panel, features = build_block_panel(swaps)
    first_block = int(swaps["block"].min())
    panel["query_time"] = ((panel["block"].to_numpy() - first_block + 1)
                           * ORDER_SLOTS_PER_BLOCK - 1)
    pool = 1 if args.pool == "v2" else 0
    sample = (panel[(panel["pool"] == pool) & (panel["next_count"] > 0)]
              .sort_values("block").reset_index(drop=True))
    cut_train, cut_val = int(len(sample) * .60), int(len(sample) * .80)
    train, validation, test = (sample.iloc[:cut_train], sample.iloc[cut_train:cut_val],
                               sample.iloc[cut_val:])
    threshold = float(train["next_markout"].quantile(args.quantile))
    y_train = train["next_markout"].to_numpy() > threshold
    y_val = validation["next_markout"].to_numpy() > threshold
    y_test = test["next_markout"].to_numpy() > threshold
    print(f"{args.pool.upper()} conditional next-block toxicity: {len(sample):,} blocks | "
          f"threshold {threshold:+.2f} bps | holdout prevalence {y_test.mean():.1%}")

    baseline = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, learning_rate=.05,
        l2_regularization=1, random_state=0)
    baseline.fit(train[features], y_train)
    base_val = baseline.predict_proba(validation[features])[:, 1]
    base_test = baseline.predict_proba(test[features])[:, 1]
    metrics("pool-state baseline", y_test, base_test)

    graph, sampler, actor_count, edge_count = build_graph(swaps, False)
    pool_node = actor_count + pool
    print(f"graph: {actor_count:,} actors | {edge_count:,} directed events")
    one = train_tgat("one-hop", sampler, pool_node, train, threshold, 1,
                     args.epochs, args.max_train)
    one_val, one_test = predict(one, pool_node, validation), predict(one, pool_node, test)
    metrics("one-hop TGAT", y_test, one_test)

    two = train_tgat("two-hop", sampler, pool_node, train, threshold, 2,
                     args.epochs, args.max_train)
    two_val, two_test = predict(two, pool_node, validation), predict(two, pool_node, test)
    metrics("two-hop TGAT", y_test, two_test)

    del graph, sampler
    shuffled_graph, shuffled_sampler, shuffled_actors, _ = build_graph(swaps, True)
    shuffled = train_tgat("actor-shuffled", shuffled_sampler,
                          shuffled_actors + pool, train, threshold, 2,
                          args.epochs, args.max_train)
    shuffled_val = predict(shuffled, shuffled_actors + pool, validation)
    shuffled_test = predict(shuffled, shuffled_actors + pool, test)
    metrics("actor-shuffled TGAT", y_test, shuffled_test)

    # The combiner is fit only on the middle validation period.  Therefore an
    # incremental result cannot be attributed to fitting the final holdout.
    combiner = LogisticRegression(C=1.0).fit(
        np.column_stack([base_val, two_val]), y_val)
    combined = combiner.predict_proba(np.column_stack([base_test, two_test]))[:, 1]
    metrics("baseline + TGAT", y_test, combined)
    print(f"combiner coefficients: pool state {combiner.coef_[0, 0]:+.3f}, "
          f"TGAT {combiner.coef_[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
