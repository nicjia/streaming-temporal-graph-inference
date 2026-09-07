"""
Ablations and typed prediction for the corporate link model.

Two experiments on top of corporate_link_prediction.py:

  1. Ablation sweep. Retrain under one fixed budget removing one component at a
     time (relation types, the second hop, the continuous-time clock) to see
     which parts carry the result. Includes a time-shuffled control.

  2. Typed head. On top of the pair embeddings, predict the relation type
     (supplier, competitor, partner, ...). Reported as per-type recall, since
     the type mix is ~95% two types and overall accuracy would hide the rest.

Trains each variant once and checkpoints it so a re-run is free.

    python benchmarks/corporate_link_variants.py
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
import torch.nn as nn  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402
from ingestion.extraction_schema import Relation  # noqa: E402

# Low-byte relation type -> dense id and readable name, for the typed head.
REL_NAMES = {int(Relation.SUPPLIES_TO): "supplier", int(Relation.OWNS): "owns",
             int(Relation.COMPETES_WITH): "competitor",
             int(Relation.PARTNERS_WITH): "partner",
             int(Relation.LENDS_TO): "lender", 20: "interlock", 21: "supply_chain"}


def ranking_metrics(positive, negatives):
    beaten = ((negatives > positive[:, None]).sum(axis=1)
              + 0.5 * (negatives == positive[:, None]).sum(axis=1))
    return {"mrr": float((1.0 / (beaten + 1)).mean()),
            "recall@1": float((beaten < 0.5).mean()),
            "recall@10": float((beaten < 10).mean()),
            "auc": roc_auc(positive, negatives.ravel())}


def load_graph(path):
    rows = []
    with gzip.open(path, "rt") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["ts"])
    ids = {}
    def nid(t): return ids.setdefault(t, len(ids))
    src = np.array([nid(r["src"]) for r in rows], dtype=np.uint32)
    dst = np.array([nid(r["dst"]) for r in rows], dtype=np.uint32)
    ts = np.array([r["ts"] for r in rows], dtype=np.uint32)
    # Low byte only: the relation TYPE (0-21). The packed code also carried
    # source and confidence in high bits, which a type embedding should not see.
    rel = np.array([r["relation_code"] & 0xFF for r in rows], dtype=np.uint16)
    return src, dst, ts, rel, len(ids)


def build(src, dst, ts, rel, vertices, shuffle_time=False, seed=0):
    if shuffle_time:
        # Reassign timestamps at random (same multiset of times). This is a
        # leakage check: a test-window edge can now get an early time and show up
        # in an earlier neighbourhood query, so the variant sees the future and
        # scores higher than the causal run. The gap measures that leak.
        t = ts.copy()
        np.random.default_rng(seed).shuffle(t)
        order = np.argsort(t, kind="stable")
        src, dst, ts, rel = src[order], dst[order], t[order], rel[order]
    g = graph_engine.PCSRGraph(vertices, int(len(src) * 2.5),
                               int(len(src) * 2.5) * 8 * 4 + (1 << 26))
    g.insert_edges(src, dst, ts, rel)
    return g, PCSRTemporalSampler(g)


def train_link(model, tr, candidates, prior, epochs, batch, rng, ckpt, reuse):
    if reuse and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print(f"  loaded {ckpt}", flush=True)
        return
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_s, tr_d, tr_t = tr
    for epoch in range(epochs):
        order = rng.permutation(len(tr_s))
        losses = []
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            neg = rng.choice(candidates, size=len(b), p=prior).astype(np.uint32)
            opt.zero_grad()
            loss, _, _ = model.loss(tr_s[b], tr_d[b], tr_t[b], neg)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        print(f"  epoch {epoch+1}: loss {np.mean(losses):.4f}", flush=True)
    torch.save(model.state_dict(), ckpt)


def eval_link(model, test, candidates, prior, negatives, rng, max_test):
    ts_, td, tt, is_new = test
    idx = np.where(is_new)[0]
    if len(idx) > max_test:
        idx = rng.choice(idx, max_test, replace=False)
    s, d, t = ts_[idx], td[idx], tt[idx]
    neg = rng.choice(candidates, size=(len(idx), negatives), p=prior).astype(np.uint32)
    pos = model.score(s, d, t).detach().numpy()
    negs = model.score_against(s, t, neg).detach().numpy()
    return ranking_metrics(pos, negs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/edgar/graph/unified_edges.jsonl.gz")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--negatives", type=int, default=50)
    ap.add_argument("--node-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--max-test", type=int, default=4000)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    src, dst, ts, rel, vertices = load_graph(args.graph)
    n_rel = int(rel.max()) + 1
    print(f"graph: {len(src):,} edges, {vertices:,} firms, {n_rel} relation codes")
    cut_tr, cut_val = int(0.70 * len(src)), int(0.85 * len(src))

    seen = set(zip(src[:cut_tr].tolist(), dst[:cut_tr].tolist()))
    is_new = np.fromiter(((int(a), int(b)) not in seen
                          for a, b in zip(src[cut_val:], dst[cut_val:])),
                         bool, len(src) - cut_val)
    deg = np.bincount(dst[:cut_tr], minlength=vertices).astype(np.float64)
    candidates = np.where(deg > 0)[0]
    prior = deg[candidates] / deg[candidates].sum()
    tr = (src[:cut_tr], dst[:cut_tr], ts[:cut_tr])
    test = (src[cut_val:], dst[cut_val:], ts[cut_val:], is_new)

    base = args.graph.replace(".jsonl.gz", "")
    variants = [
        ("topology + time (2-hop)", dict(num_layers=2)),
        ("+ relation types", dict(num_layers=2, num_relations=n_rel)),
        ("one hop only", dict(num_layers=1, num_relations=n_rel)),
        ("time shuffled (control)", dict(num_layers=2, num_relations=n_rel,
                                         _shuffle=True)),
    ]

    print(f"\n{'='*62}\nABLATION: what drives next-link prediction (new links)\n{'='*62}")
    print(f"{'variant':<28}{'MRR':>8}{'R@10':>8}{'AUC':>8}")
    print(f"{'adamic-adar (baseline)':<28}{0.128:>8.3f}{0.264:>8.3f}{0.564:>8.3f}")
    results = {}
    for name, cfg in variants:
        shuffle = cfg.pop("_shuffle", False)
        g, sampler = build(src, dst, ts, rel, vertices, shuffle_time=shuffle)
        model = TGATLinkModel(vertices, sampler, node_dim=args.node_dim,
                              time_dim=args.node_dim, **cfg)
        tag = name.split()[0].replace("+", "rel")
        ckpt = f"{base}.var_{tag}_{'shuf' if shuffle else 'real'}.pt"
        print(f"\n[{name}]", flush=True)
        train_link(model, tr, candidates, prior, args.epochs, args.batch, rng,
                   ckpt, args.reuse)
        m = eval_link(model, test, candidates, prior, args.negatives, rng, args.max_test)
        results[name] = (m, model, g, sampler)
        print(f"{name:<28}{m['mrr']:>8.3f}{m['recall@10']:>8.3f}{m['auc']:>8.3f}")

    print(f"\n{'='*62}\nSUMMARY\n{'='*62}")
    print(f"{'variant':<28}{'MRR':>8}{'R@10':>8}{'AUC':>8}")
    print(f"{'adamic-adar (baseline)':<28}{0.128:>8.3f}{0.264:>8.3f}{0.564:>8.3f}")
    for name, (m, *_ ) in results.items():
        print(f"{name:<28}{m['mrr']:>8.3f}{m['recall@10']:>8.3f}{m['auc']:>8.3f}")

    # ---- typed head: predict the relation TYPE of a forming edge -----------
    print(f"\n{'='*62}\nTYPED PREDICTION: what kind of relationship forms\n{'='*62}")
    _, best_model, _, _ = results["+ relation types"]
    enc = best_model.encoder
    type_ids = {c: i for i, c in enumerate(sorted(REL_NAMES))}
    inv = {i: REL_NAMES[c] for c, i in type_ids.items()}

    def embed_pairs(s, d, t):
        with torch.no_grad():
            both = enc(np.concatenate([s, d]).astype(np.int64),
                       np.concatenate([t, t]).astype(np.int64))
        sh, dh = both.chunk(2, dim=0)
        return torch.cat([sh, dh], dim=-1)

    head = nn.Sequential(nn.Linear(args.node_dim * 2, args.node_dim), nn.ReLU(),
                         nn.Linear(args.node_dim, len(type_ids)))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    tr_rel = rel[:cut_tr]
    keep = np.isin(tr_rel, list(type_ids))
    ks, kd, kt, ky = (src[:cut_tr][keep], dst[:cut_tr][keep], ts[:cut_tr][keep],
                      np.array([type_ids[int(r)] for r in tr_rel[keep]]))
    # Class-balanced loss: the rare text-derived types would vanish otherwise.
    counts = np.bincount(ky, minlength=len(type_ids)).astype(np.float64)
    cw = torch.tensor(1.0 / np.clip(counts, 1, None), dtype=torch.float32)
    cw = cw / cw.sum() * len(type_ids)
    lossfn = nn.CrossEntropyLoss(weight=cw)
    print(f"training typed head on {len(ks):,} edges, {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        order = rng.permutation(len(ks))
        losses = []
        for i in range(0, len(order), 512):
            b = order[i:i + 512]
            feats = embed_pairs(ks[b], kd[b], kt[b])
            opt.zero_grad()
            loss = lossfn(head(feats), torch.tensor(ky[b]))
            loss.backward(); opt.step(); losses.append(float(loss))
        print(f"  epoch {epoch+1}: loss {np.mean(losses):.4f}", flush=True)

    te_rel = rel[cut_val:]
    tkeep = np.isin(te_rel, list(type_ids)) & is_new
    es, ed, et = src[cut_val:][tkeep], dst[cut_val:][tkeep], ts[cut_val:][tkeep]
    ey = np.array([type_ids[int(r)] for r in te_rel[tkeep]])
    with torch.no_grad():
        pred = head(embed_pairs(es, ed, et)).argmax(1).numpy()
    print(f"\n{'relation type':<16}{'support':>8}{'recall':>8}")
    for i in range(len(type_ids)):
        mask = ey == i
        if mask.sum():
            print(f"{inv[i]:<16}{int(mask.sum()):>8}{(pred[mask]==i).mean():>8.2f}")
    macro = np.mean([(pred[ey == i] == i).mean() for i in range(len(type_ids))
                     if (ey == i).sum()])
    print(f"{'macro-recall':<16}{'':>8}{macro:>8.2f}   (random = {1/len(type_ids):.2f})")


if __name__ == "__main__":
    main()
