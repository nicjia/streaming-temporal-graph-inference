"""Cold-start split of the corporate link prediction task.

Common-neighbours and Adamic-Adar score a pair only through shared neighbours, so
on a pair with no common neighbour in the training window they return zero and
rank no better than chance. This splits held-out new links by whether the true
pair shares a training common neighbour and scores each split separately against
the same degree-matched negatives, to see where the learned embeddings help.

Reuses the +relations checkpoint from corporate_link_variants.py.

    python benchmarks/link_prediction_coldstart.py
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
import torch  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402


def ranking_metrics(pos, negs):
    beaten = ((negs > pos[:, None]).sum(1) + 0.5 * (negs == pos[:, None]).sum(1))
    return dict(mrr=float((1 / (beaten + 1)).mean()),
                recall1=float((beaten < 0.5).mean()),
                recall10=float((beaten < 10).mean()),
                auc=roc_auc(pos, negs.ravel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/edgar/graph/unified_edges.jsonl.gz")
    ap.add_argument("--negatives", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-test", type=int, default=6000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    with gzip.open(args.graph, "rt") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["ts"])
    ids = {}
    nid = lambda t: ids.setdefault(t, len(ids))
    src = np.array([nid(r["src"]) for r in rows], np.uint32)
    dst = np.array([nid(r["dst"]) for r in rows], np.uint32)
    ts = np.array([r["ts"] for r in rows], np.uint32)
    rel = np.array([r["relation_code"] & 0xFF for r in rows], np.uint16)
    V = len(ids)
    n_rel = int(rel.max()) + 1
    cut_tr, cut_val = int(0.70 * len(src)), int(0.85 * len(src))
    print(f"graph: {len(src):,} edges, {V:,} firms, {n_rel} relation types")

    seen = set(zip(src[:cut_tr].tolist(), dst[:cut_tr].tolist()))
    ts_src, ts_dst, ts_t = src[cut_val:], dst[cut_val:], ts[cut_val:]
    is_new = np.fromiter(((int(a), int(b)) not in seen
                          for a, b in zip(ts_src, ts_dst)), bool, len(ts_src))

    # Training adjacency (undirected) for the classical scores and the split.
    adj = defaultdict(set)
    for a, b in zip(src[:cut_val].tolist(), dst[:cut_val].tolist()):
        adj[a].add(b); adj[b].add(a)
    deg = np.bincount(dst[:cut_tr], minlength=V).astype(np.float64)
    cand = np.where(deg > 0)[0]
    prior = deg[cand] / deg[cand].sum()

    def score_adamic(s, d):
        shared = adj[s] & adj[d]
        return float(sum(1.0 / np.log(len(adj[n]) + 2) for n in shared))

    def score_common(s, d):
        return float(len(adj[s] & adj[d]))

    # Build graph with relation types and load/train the +relations model.
    g = graph_engine.PCSRGraph(V, int(len(src) * 2.5),
                               int(len(src) * 2.5) * 8 * 4 + (1 << 26))
    g.insert_edges(src, dst, ts, rel)
    sampler = PCSRTemporalSampler(g)
    model = TGATLinkModel(V, sampler, node_dim=64, time_dim=64,
                          num_layers=2, num_relations=n_rel)
    ckpt = args.graph.replace(".jsonl.gz", ".var_rel_real.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print(f"loaded {ckpt}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trs, trd, trt = src[:cut_tr], dst[:cut_tr], ts[:cut_tr]
        for ep in range(args.epochs):
            o = rng.permutation(len(trs))
            for i in range(0, len(o), 200):
                b = o[i:i + 200]
                neg = rng.choice(cand, size=len(b), p=prior).astype(np.uint32)
                opt.zero_grad()
                loss, _, _ = model.loss(trs[b], trd[b], trt[b], neg)
                loss.backward(); opt.step()
        torch.save(model.state_dict(), ckpt)

    # Split new links by whether the true pair shares a training neighbour.
    new_idx = np.where(is_new)[0]
    has_cn = np.fromiter((len(adj[int(ts_src[i])] & adj[int(ts_dst[i])]) > 0
                          for i in new_idx), bool, len(new_idx))
    print(f"\nnew held-out links: {len(new_idx):,}")
    print(f"  no common neighbour (cold): {(~has_cn).sum():,} ({(~has_cn).mean():.0%})")
    print(f"  >=1 common neighbour (warm): {has_cn.sum():,} ({has_cn.mean():.0%})")

    for label, mask in (("COLD (no common neighbour)", ~has_cn),
                        ("WARM (>=1 common neighbour)", has_cn)):
        idx = new_idx[mask]
        if len(idx) > args.max_test:
            idx = rng.choice(idx, args.max_test, replace=False)
        if len(idx) == 0:
            continue
        s, d, t = ts_src[idx], ts_dst[idx], ts_t[idx]
        neg = rng.choice(cand, size=(len(idx), args.negatives),
                         p=prior).astype(np.uint32)
        print(f"\n=== {label}  ({len(idx):,} queries) ===")
        print(f"{'scorer':<20}{'MRR':>8}{'R@1':>8}{'R@10':>8}{'AUC':>8}")
        for name, fn in (("common-neighbours", score_common),
                         ("adamic-adar", score_adamic)):
            pos = np.array([fn(int(s[i]), int(d[i])) for i in range(len(idx))])
            negs = np.stack([[fn(int(s[i]), int(neg[i, k]))
                              for k in range(args.negatives)] for i in range(len(idx))])
            m = ranking_metrics(pos, negs)
            print(f"{name:<20}{m['mrr']:>8.3f}{m['recall1']:>8.3f}"
                  f"{m['recall10']:>8.3f}{m['auc']:>8.3f}")
        pos = model.score(s, d, t).detach().numpy()
        negs = model.score_against(s, t, neg).detach().numpy()
        m = ranking_metrics(pos, negs)
        print(f"{'TGAT':<20}{m['mrr']:>8.3f}{m['recall1']:>8.3f}"
              f"{m['recall10']:>8.3f}{m['auc']:>8.3f}")


if __name__ == "__main__":
    main()
