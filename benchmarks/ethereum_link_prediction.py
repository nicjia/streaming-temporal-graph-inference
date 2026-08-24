"""
Can the TGAT predict who an Ethereum address transacts with next?

The headline split is between two very different questions:

  REPEAT links -- the pair has transacted before. Mostly trivial: addresses use
  the same router, the same stablecoin, the same bridge, over and over. A
  frequency counter is very hard to beat here and a model that only wins on
  repeats has learned nothing worth having.

  NEW links -- the pair has never transacted. This is protocol adoption and
  capital migration: which contract will this wallet start using? Frequency and
  recency cannot answer it *by construction*, since there is no history for the
  pair. Any signal has to come from the structure and timing of the wallet's
  other activity, which is exactly what a temporal graph network is for.

Everything is scored against degree-matched negatives. With uniform negatives
this task degenerates into "is this a popular contract", which on GDELT scored
0.915 AUC for a single popularity number and beat the network outright.
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402
from ingestion.ethereum import load_transactions  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402


def cached_globs():
    root = os.path.expanduser("~/.cache/huggingface/hub")
    base = f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet/snapshots/*"
    tx = sorted(glob.glob(f"{base}/transactions/*.parquet"))
    if not tx:
        raise SystemExit("no cached shards")
    return os.path.dirname(tx[0]) + "/*.parquet", os.path.dirname(tx[0]).replace("transactions", "blocks") + "/*.parquet"


def ranking_metrics(positive, negatives):
    """
    MRR and Recall@k of the true target against its sampled negatives.

    Ties take the mid-rank. This is not a detail: a baseline that cannot score
    the case at all -- pair frequency on a link whose pair has never been seen
    -- returns zero for the positive and zero for every negative, and a strict
    `>` comparison reads that as the positive winning outright. Measured, that
    reported MRR 0.850 for a scorer whose AUC was 0.489. Ties are ignorance,
    not skill, and must be scored as a coin flip.
    """
    beaten = ((negatives > positive[:, None]).sum(axis=1)
              + 0.5 * (negatives == positive[:, None]).sum(axis=1))
    return {
        "mrr": float((1.0 / (beaten + 1)).mean()),
        "recall@1": float((beaten < 0.5).mean()),
        "recall@10": float((beaten < 10).mean()),
        "auc": roc_auc(positive, negatives.ravel()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-degree", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--max-test", type=int, default=4000)
    parser.add_argument("--train-events", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--checkpoint", default="/tmp/eth_tgat.pt")
    parser.add_argument("--reuse", action="store_true",
                        help="Load the checkpoint instead of retraining")
    args = parser.parse_args()

    tx_glob, block_glob = cached_globs()
    table, addresses = load_transactions(tx_glob, block_glob, min_degree=args.min_degree)
    src = table["src"].to_numpy(np.int64)
    dst = table["dst"].to_numpy(np.int64)
    ts = table["ts"].to_numpy(np.int64)
    rel = table["relation"].to_numpy(np.uint16)
    vertices = len(addresses)

    # Temporal split. Chronological, never random: a random split lets the
    # model train on the future of the very pairs it is tested on.
    cut_train = int(len(src) * 0.75)
    cut_val = int(len(src) * 0.85)
    print(f"\nsplit: train {cut_train:,} | val {cut_val-cut_train:,} | test {len(src)-cut_val:,}")

    # Build the full graph once. The sampler only reads edges strictly before a
    # query time, so this is causal; what must be controlled is the weights.
    graph = graph_engine.PCSRGraph(vertices, int(len(src) * 2.5), int(len(src) * 2.5) * 8 * 4 + (1 << 26))
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32),
                       ts.astype(np.uint32), rel)
    sampler = PCSRTemporalSampler(graph)
    print(f"graph: {graph.num_edges:,} edges, {vertices:,} vertices, "
          f"chronological: {sampler.validate()}")

    # Which pairs exist in the training window -- defines repeat vs new.
    seen_pairs = set(zip(src[:cut_train].tolist(), dst[:cut_train].tolist()))
    test_src, test_dst, test_ts = src[cut_val:], dst[cut_val:], ts[cut_val:]
    is_new = np.fromiter(((s, d) not in seen_pairs for s, d in zip(test_src, test_dst)),
                         dtype=bool, count=len(test_src))
    print(f"test edges: {(~is_new).sum():,} repeat ({(~is_new).mean():.0%}), "
          f"{is_new.sum():,} new ({is_new.mean():.0%})")

    # Degree-matched negative sampler, from the training window only.
    in_degree = np.bincount(dst[:cut_train], minlength=vertices).astype(np.float64)
    candidates = np.flatnonzero(in_degree > 0)
    prior = in_degree[candidates] / in_degree[candidates].sum()

    # ---- baselines, all fitted on the training window ---------------------
    pair_count = defaultdict(int)
    last_seen = {}
    partner_count = defaultdict(lambda: defaultdict(int))
    for s, d, t in zip(src[:cut_train], dst[:cut_train], ts[:cut_train]):
        pair_count[(s, d)] += 1
        last_seen[(s, d)] = t
        partner_count[s][d] += 1
    neighbors_of = {s: set(p) for s, p in partner_count.items()}
    in_neighbors = defaultdict(set)
    for s, d in zip(src[:cut_train], dst[:cut_train]):
        in_neighbors[d].add(s)
    popularity = np.log1p(in_degree)

    def score_popularity(s, d, t):
        return popularity[d]

    def score_pair_frequency(s, d, t):
        return np.log1p(pair_count.get((s, d), 0))

    def score_recency(s, d, t):
        seen = last_seen.get((s, d))
        return 0.0 if seen is None else 1.0 / (1.0 + (t - seen) / 86400.0)

    def score_adamic_adar(s, d, t):
        # Shared counterparties, down-weighted by how indiscriminate they are.
        common = neighbors_of.get(s, set()) & in_neighbors.get(d, set())
        return float(sum(1.0 / np.log(2.0 + in_degree[c]) for c in common))

    control = np.random.default_rng(1)

    baselines = {
        "control: random": lambda s, d, t: float(control.random()),
        "popularity (in-degree)": score_popularity,
        "pair frequency": score_pair_frequency,
        "pair recency": score_recency,
        "adamic-adar": score_adamic_adar,
        "frequency + popularity": lambda s, d, t: score_pair_frequency(s, d, t) * 3 + popularity[d],
    }

    # ---- sample the evaluation set ---------------------------------------
    rng = np.random.default_rng(0)

    def take(mask, limit):
        idx = np.flatnonzero(mask)
        if len(idx) > limit:
            idx = idx[np.linspace(0, len(idx) - 1, limit).astype(int)]
        return idx

    subsets = {"repeat": take(~is_new, args.max_test), "new": take(is_new, args.max_test)}
    negatives = {k: rng.choice(candidates, size=(len(v), args.negatives), p=prior)
                 for k, v in subsets.items()}

    # ---- train the TGAT ---------------------------------------------------
    torch.manual_seed(0)
    num_relations = int(rel.max()) + 1
    model = TGATLinkModel(vertices, sampler, node_dim=args.node_dim,
                          time_dim=args.node_dim, num_layers=args.layers,
                          num_neighbors=args.neighbors, num_relations=num_relations)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = max(1, cut_train // args.train_events)
    tr_s, tr_d, tr_t = src[:cut_train:step], dst[:cut_train:step], ts[:cut_train:step]
    print(f"\ntraining on {len(tr_s):,} sampled events, {args.epochs} epochs, "
          f"{num_relations} relation types")

    if args.reuse and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"  loaded {args.checkpoint}, skipping training")
        args.epochs = 0

    model.train()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        for off in range(0, len(tr_s), 256):
            batch = slice(off, min(off + 256, len(tr_s)))
            if batch.stop - batch.start < 8:
                continue
            n = batch.stop - batch.start
            optimizer.zero_grad()
            loss, _, _ = model.loss(tr_s[batch], tr_d[batch], tr_t[batch],
                                    rng.choice(candidates, n, p=prior),
                                    rng.choice(candidates, n, p=prior))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"  epoch {epoch+1}: loss {np.mean(losses):.4f} "
              f"({time.perf_counter()-started:.0f}s)")

    if args.epochs:
        torch.save(model.state_dict(), args.checkpoint)
        print(f"  saved {args.checkpoint}")

    # ---- evaluate ---------------------------------------------------------
    model.eval()
    header = f"{'scorer':<26}{'MRR':>8}{'R@1':>8}{'R@10':>8}{'AUC':>8}"
    for label, idx in subsets.items():
        s, d, t = test_src[idx], test_dst[idx], test_ts[idx]
        neg = negatives[label]
        print(f"\n=== {label.upper()} links ({len(idx):,} test edges, "
              f"{args.negatives} degree-matched negatives) ===")
        print(header); print("-" * len(header))

        for name, fn in baselines.items():
            pos = np.array([fn(a, b, c) for a, b, c in zip(s, d, t)])
            ng = np.array([[fn(a, nb, c) for nb in row] for a, row, c in zip(s, neg, t)])
            m = ranking_metrics(pos, ng)
            print(f"{name:<26}{m['mrr']:>8.3f}{m['recall@1']:>8.3f}"
                  f"{m['recall@10']:>8.3f}{m['auc']:>8.3f}")

        with torch.no_grad():
            pos = model.score(s, d, t).numpy()
            ng = np.stack([model.score(s, neg[:, k], t).numpy()
                           for k in range(args.negatives)], axis=1)
        m = ranking_metrics(pos, ng)
        print(f"{'TGAT':<26}{m['mrr']:>8.3f}{m['recall@1']:>8.3f}"
              f"{m['recall@10']:>8.3f}{m['auc']:>8.3f}")


if __name__ == "__main__":
    main()
