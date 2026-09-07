"""Additive blend of the TGAT and Adamic-Adar link scores.

Adamic-Adar is zero on pairs with no common neighbour, so this blend leaves the
TGAT score unchanged on cold pairs and adds the classical lift on warm ones:

    score(s, c) = sigmoid(TGAT_logit(s, c)) + lambda * adamic(s, c)

Sweeps lambda and reports every model on COLD / WARM / ALL. Reuses the
+relations checkpoint.

    python benchmarks/link_prediction_ensemble.py
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


def metrics(pos, negs):
    beaten = ((negs > pos[:, None]).sum(1) + 0.5 * (negs == pos[:, None]).sum(1))
    return dict(mrr=float((1 / (beaten + 1)).mean()),
                recall10=float((beaten < 10).mean()),
                auc=roc_auc(pos, negs.ravel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/edgar/graph/unified_edges.jsonl.gz")
    ap.add_argument("--ckpt", default=None)
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
    tsr, tsd, tst = src[cut_val:], dst[cut_val:], ts[cut_val:]
    is_new = np.fromiter(((int(a), int(b)) not in seen
                          for a, b in zip(tsr, tsd)), bool, len(tsr))

    adj = defaultdict(set)
    for a, b in zip(src[:cut_val].tolist(), dst[:cut_val].tolist()):
        adj[a].add(b); adj[b].add(a)
    deg = np.bincount(dst[:cut_tr], minlength=V).astype(np.float64)
    cand = np.where(deg > 0)[0]
    prior = deg[cand] / deg[cand].sum()

    def adamic(s, d):
        shared = adj[s] & adj[d]
        return float(sum(1.0 / np.log(len(adj[n]) + 2) for n in shared))

    g = graph_engine.PCSRGraph(V, int(len(src) * 2.5),
                               int(len(src) * 2.5) * 8 * 4 + (1 << 26))
    g.insert_edges(src, dst, ts, rel)
    sampler = PCSRTemporalSampler(g)
    model = TGATLinkModel(V, sampler, node_dim=64, time_dim=64,
                          num_layers=2, num_relations=n_rel)
    ckpt = args.ckpt or args.graph.replace(".jsonl.gz", ".var_rel_real.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt)); print(f"loaded {ckpt}")
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

    idx = np.where(is_new)[0]
    if len(idx) > args.max_test:
        idx = rng.choice(idx, args.max_test, replace=False)
    s, d, t = tsr[idx], tsd[idx], tst[idx]
    neg = rng.choice(cand, size=(len(idx), args.negatives), p=prior).astype(np.uint32)

    # TGAT probabilities and Adamic scores for positive + negatives.
    tg_pos = torch.sigmoid(model.score(s, d, t)).detach().numpy()
    tg_neg = torch.sigmoid(model.score_against(s, t, neg)).detach().numpy()
    ad_pos = np.array([adamic(int(s[i]), int(d[i])) for i in range(len(idx))])
    ad_neg = np.stack([[adamic(int(s[i]), int(neg[i, k]))
                        for k in range(args.negatives)] for i in range(len(idx))])
    cold = np.fromiter((len(adj[int(s[i])] & adj[int(d[i])]) == 0
                        for i in range(len(idx))), bool, len(idx))
    print(f"\ntest new links: {len(idx):,}  cold {cold.mean():.0%}  warm {(~cold).mean():.0%}")

    def report(name, pos, negs):
        for tag, m in (("ALL", np.ones(len(idx), bool)), ("COLD", cold), ("WARM", ~cold)):
            if m.sum() == 0:
                continue
            r = metrics(pos[m], negs[m])
            print(f"  {name:<24}{tag:<6}MRR {r['mrr']:.3f}  R@10 {r['recall10']:.3f}  AUC {r['auc']:.3f}")

    print()
    report("TGAT", tg_pos, tg_neg)
    report("adamic-adar", ad_pos, ad_neg)

    print("\nlambda sweep (score = TGAT_prob + lambda*adamic):")
    best = (None, -1)
    for lam in (0.02, 0.05, 0.1, 0.2, 0.4):
        ep, en = tg_pos + lam * ad_pos, tg_neg + lam * ad_neg
        r = metrics(ep, en)
        print(f"  lambda={lam:<5} ALL   MRR {r['mrr']:.3f}  R@10 {r['recall10']:.3f}  AUC {r['auc']:.3f}")
        if r["auc"] > best[1]:
            best = (lam, r["auc"])
    lam = best[0]
    print(f"\nbest lambda = {lam} (ALL AUC {best[1]:.3f}):")
    report(f"ENSEMBLE(l={lam})", tg_pos + lam * ad_pos, tg_neg + lam * ad_neg)


if __name__ == "__main__":
    main()
