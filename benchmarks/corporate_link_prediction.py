"""
Next-relationship prediction on the corporate graph.

Given the network state at time t, rank the firm each source connects to next
against degree-matched negatives. Baselines: popularity, common neighbours,
Adamic-Adar. New pairs (unseen in training) are reported separately from repeat
pairs. Reads the assembled graph from data/edgar/graph/unified_edges.jsonl.gz.

    python benchmarks/corporate_link_prediction.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, ROOT)

import graph_engine  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402


def ranking_metrics(positive, negatives):
    """MRR / Recall@k / AUC of the true target against its negatives.

    Ties take the mid-rank so an unscoreable case (positive and negatives all
    zero) is not counted as a win.
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
    parser.add_argument("--graph", default="data/edgar/graph/unified_edges.jsonl.gz")
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--max-test", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load the assembled graph and map tickers to dense ids.
    rows = []
    with gzip.open(args.graph, "rt") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["ts"])
    ids = {}
    def nid(t):
        return ids.setdefault(t, len(ids))
    src = np.array([nid(r["src"]) for r in rows], dtype=np.uint32)
    dst = np.array([nid(r["dst"]) for r in rows], dtype=np.uint32)
    ts = np.array([r["ts"] for r in rows], dtype=np.uint32)
    rel = np.array([r["relation_code"] for r in rows], dtype=np.uint16)
    wt = np.array([r["weight"] for r in rows], dtype=np.float32)
    vertices = len(ids)
    print(f"graph: {len(rows):,} edges, {vertices:,} firms")

    # Chronological split: a random split would leak a node's future state.
    cut_tr = int(0.70 * len(rows))
    cut_val = int(0.85 * len(rows))
    print(f"split: train {cut_tr:,} | val {cut_val-cut_tr:,} | test {len(rows)-cut_val:,}")

    graph = graph_engine.PCSRGraph(vertices, int(len(rows) * 2.5),
                                   int(len(rows) * 2.5) * 8 * 4 + (1 << 26),
                                   store_weights=True)
    graph.insert_edges(src, dst, ts, rel, wt)
    sampler = PCSRTemporalSampler(graph)

    test_src, test_dst, test_t = src[cut_val:], dst[cut_val:], ts[cut_val:]

    # New vs repeat: a pair seen in training is memorisation; a new pair is
    # formation, which no pair-history baseline can score.
    seen = set(zip(src[:cut_tr].tolist(), dst[:cut_tr].tolist()))
    is_new = np.fromiter(((int(s), int(d)) not in seen
                          for s, d in zip(test_src, test_dst)), bool, len(test_src))
    print(f"test edges: {(~is_new).sum():,} repeat ({(~is_new).mean():.0%}), "
          f"{is_new.sum():,} new ({is_new.mean():.0%})")

    # Degree-matched negatives from the training window; uniform sampling would
    # let "is this an active firm at all" separate most pairs on its own.
    deg = np.bincount(dst[:cut_tr], minlength=vertices).astype(np.float64)
    candidates = np.where(deg > 0)[0]
    prior = deg[candidates] / deg[candidates].sum()

    # Classical structural baselines.
    train_adj = defaultdict(set)
    for s, d in zip(src[:cut_val].tolist(), dst[:cut_val].tolist()):
        train_adj[s].add(d)
        train_adj[d].add(s)
    popularity = deg.copy()

    def score_popularity(s, d, t):
        return popularity[d]

    def score_common(s, d, t):
        return float(len(train_adj[s] & train_adj[d]))

    def score_adamic(s, d, t):
        shared = train_adj[s] & train_adj[d]
        return float(sum(1.0 / np.log(len(train_adj[n]) + 2) for n in shared))

    baselines = {"popularity": score_popularity,
                 "common-neighbours": score_common,
                 "adamic-adar": score_adamic}

    # Train the TGAT link model on sampled training events.
    model = TGATLinkModel(vertices, sampler, node_dim=args.node_dim,
                          time_dim=args.node_dim)
    import torch
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_s, tr_d, tr_t = src[:cut_tr], dst[:cut_tr], ts[:cut_tr]
    ckpt = args.graph.replace(".jsonl.gz", ".tgat.pt")
    if args.reuse and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt)); args.epochs = 0
        print(f"\nloaded trained model from {ckpt}", flush=True)
    print(f"\ntraining {len(tr_s):,} events, {args.epochs} epochs (CPU)...", flush=True)
    for epoch in range(args.epochs):
        order = rng.permutation(len(tr_s))
        losses = []
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            neg = rng.choice(candidates, size=len(b), p=prior).astype(np.uint32)
            optimizer.zero_grad()
            loss, _, _ = model.loss(tr_s[b], tr_d[b], tr_t[b], neg)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        print(f"  epoch {epoch+1}: loss {np.mean(losses):.4f}", flush=True)
    if args.epochs:
        torch.save(model.state_dict(), ckpt)

    # Evaluate new and repeat separately.
    def take(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        return idx

    subsets = {"new": take(is_new, args.max_test),
               "repeat": take(~is_new, args.max_test)}

    for label, idx in subsets.items():
        if len(idx) == 0:
            continue
        s = test_src[idx]; d = test_dst[idx]; t = test_t[idx]
        neg = rng.choice(candidates, size=(len(idx), args.negatives),
                         p=prior).astype(np.uint32)
        print(f"\n=== {label} links ({len(idx):,}, {args.negatives} "
              f"degree-matched negatives) ===")
        print(f"{'scorer':<22}{'MRR':>8}{'R@1':>8}{'R@10':>8}{'AUC':>8}")

        for name, fn in baselines.items():
            pos = np.array([fn(int(s[i]), int(d[i]), int(t[i])) for i in range(len(idx))])
            negs = np.stack([[fn(int(s[i]), int(neg[i, k]), int(t[i]))
                              for k in range(args.negatives)] for i in range(len(idx))])
            m = ranking_metrics(pos, negs)
            print(f"{name:<22}{m['mrr']:>8.3f}{m['recall@1']:>8.3f}"
                  f"{m['recall@10']:>8.3f}{m['auc']:>8.3f}")

        pos = model.score(s, d, t).detach().numpy()
        negs = model.score_against(s, t, neg).detach().numpy()
        m = ranking_metrics(pos, negs)
        print(f"{'TGAT':<22}{m['mrr']:>8.3f}{m['recall@1']:>8.3f}"
              f"{m['recall@10']:>8.3f}{m['auc']:>8.3f}")


if __name__ == "__main__":
    main()
