"""
Does this TGAT reproduce published numbers on the standard benchmarks?

Every other study in this repo asks a question nobody knows the answer to,
which makes them impossible to check. This one asks a question the literature
has already answered, which makes it the only study here that can falsify the
implementation.

Wikipedia and Reddit from the JODIE release (Kumar et al., KDD 2019) are the
datasets that TGAT, TGN, CAWN, GraphMixer and DyGFormer all report on. If this
implementation lands near the published range, the model is doing what the
paper says it does. If it lands far off, there is a bug -- and no amount of
good-looking results on Ethereum or supply chains would have revealed it.

Protocol, matched to the literature rather than chosen to flatter:
  * 70/15/15 chronological split
  * one uniformly-sampled negative destination per positive, source held fixed
  * Average Precision computed PER MINI-BATCH of 200 and averaged, which is what
    TGAT, TGN and DyGLib all actually do. A global AP pooled over the test set
    is a different statistic and does not compare to any published number.
  * transductive = all test edges

The inductive split here is deliberately not the published one and must not be
read as comparable. The literature masks 10% of nodes and deletes their training
edges to manufacture unseen nodes. This reports the nodes that are *naturally*
absent from the training window -- no edges removed, nothing synthesised. That
is a different and arguably stricter question, so it is labelled separately.

One handicap is worth stating up front: these datasets ship 172-dimensional
LIWC edge features and this engine carries a uint16 relation id per edge, not a
feature vector. Published TGAT consumes those features. This run therefore has
strictly less information than the papers it is being compared against, and
should be read as a floor on the implementation rather than a like-for-like
reproduction.

Usage:  python benchmarks/jodie_benchmark.py --dataset wikipedia
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from backtest.evaluate import average_precision, roc_auc  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402


def load(dataset):
    """JODIE csv -> a single dense id space, bipartite users then items."""
    path = os.path.join(ROOT, "data/tgb", f"{dataset}.csv")
    frame = pd.read_csv(path, usecols=[0, 1, 2], header=0,
                        names=["u", "i", "t"], dtype={"u": np.int64, "i": np.int64,
                                                      "t": np.float64})
    n_users = int(frame["u"].max()) + 1
    src = frame["u"].to_numpy(np.int64)
    dst = frame["i"].to_numpy(np.int64) + n_users
    ts = frame["t"].to_numpy(np.float64)
    order = np.argsort(ts, kind="stable")
    return src[order], dst[order], ts[order], n_users + int(frame["i"].max()) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wikipedia")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--node-dim", type=int, default=100)
    p.add_argument("--train-events", type=int, default=60000)
    p.add_argument("--eval-batch", type=int, default=200)
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()

    src, dst, ts_raw, vertices = load(args.dataset)
    edges = len(src)
    print(f"{args.dataset}: {edges:,} interactions, {vertices:,} nodes, "
          f"span {ts_raw[-1]-ts_raw[0]:,.0f}s")

    # The engine stores uint32 seconds; JODIE ships float seconds from zero.
    ts = np.clip(ts_raw, 0, None).astype(np.uint32)
    cut_train, cut_val = int(edges * 0.70), int(edges * 0.85)

    graph = graph_engine.PCSRGraph(vertices, int(edges * 2.5),
                                   int(edges * 2.5) * 8 * 4 + (1 << 26))
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32), ts,
                       np.zeros(edges, dtype=np.uint16))
    sampler = PCSRTemporalSampler(graph)
    print(f"graph: {graph.num_edges:,} edges, chronological: {sampler.validate()}")

    seen = set(src[:cut_train].tolist()) | set(dst[:cut_train].tolist())
    test = np.arange(cut_val, edges)
    touches_new = np.fromiter(
        ((int(src[i]) not in seen) or (int(dst[i]) not in seen) for i in test),
        dtype=bool, count=len(test))
    print(f"split: train {cut_train:,} | val {cut_val-cut_train:,} | "
          f"test {len(test):,}  ({touches_new.sum():,} inductive, "
          f"{touches_new.mean():.1%})")

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = TGATLinkModel(vertices, sampler, node_dim=args.node_dim,
                          time_dim=args.node_dim, num_layers=args.layers,
                          num_neighbors=args.neighbors)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ckpt = os.path.join(ROOT, f"jodie_{args.dataset}.pt")
    step = max(1, cut_train // args.train_events)
    tr_s, tr_d, tr_t = src[:cut_train:step], dst[:cut_train:step], ts[:cut_train:step]
    print(f"\ntraining on {len(tr_s):,} sampled interactions, {args.epochs} epochs")

    if args.reuse and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print(f"  loaded {ckpt}")
        args.epochs = 0

    model.train()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        for off in range(0, len(tr_s), args.batch):
            end = min(off + args.batch, len(tr_s))
            if end - off < 8:
                continue
            optimizer.zero_grad()
            loss, _, _ = model.loss(tr_s[off:end], tr_d[off:end], tr_t[off:end],
                                    rng.integers(0, vertices, end - off))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"  epoch {epoch+1}: loss {np.mean(losses):.4f} "
              f"({time.perf_counter()-started:.0f}s)")
    if args.epochs:
        torch.save(model.state_dict(), ckpt)

    # ---- evaluate: one uniform negative per positive, as published ---------
    model.eval()

    def evaluate(idx):
        """Per-batch AP/AUC, averaged. Matches TGN/DyGLib; batch size matters."""
        aps, aucs = [], []
        for off in range(0, len(idx), args.eval_batch):
            part = idx[off:off + args.eval_batch]
            if len(part) < 8:
                continue
            s_b, d_b, t_b = src[part], dst[part], ts[part]
            neg = rng.integers(0, vertices, len(part))
            with torch.no_grad():
                pos = model.score(s_b, d_b, t_b).numpy()
                ng = model.score(s_b, neg, t_b).numpy()
            aps.append(average_precision(pos, ng))
            aucs.append(roc_auc(pos, ng))
        return float(np.mean(aps)), float(np.mean(aucs)), len(aps)

    print(f"\n{'split':<34}{'edges':>9}{'AP':>9}{'AUC':>9}")
    print("-" * 61)
    for label, mask in (("transductive (comparable)", np.ones(len(test), bool)),
                        ("inductive (naturally unseen)", touches_new)):
        idx = test[mask]
        if len(idx) < args.eval_batch:
            print(f"{label:<34}{len(idx):>9,}{'n/a':>9}{'n/a':>9}")
            continue
        ap, auc, nb = evaluate(idx)
        print(f"{label:<34}{len(idx):>9,}{ap:>9.4f}{auc:>9.4f}")

    print("\nPublished transductive AP, same protocol (1 uniform negative, per-batch):")
    print("  wikipedia   TGAT 0.9534 (Xu 2020) / 0.9694 (DyGLib)   TGN 0.9845")
    print("  reddit      TGAT 0.9812 (Xu 2020) / 0.9852 (DyGLib)   TGN 0.9863")
    print("  Those models consume the 172-dim LIWC edge features; this engine")
    print("  carries a uint16 relation id and does not. Node features do not")
    print("  exist for these datasets -- every framework feeds zeros -- so that")
    print("  is not a difference. Read this as a floor, not a reproduction.")


if __name__ == "__main__":
    main()
