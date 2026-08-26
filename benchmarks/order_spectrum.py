"""
At what timescale does edge ordering actually carry information?

There is a live question in the temporal-graph literature about whether the
standard benchmarks are temporal at all. "What Do Temporal Graph Learning
Models Learn?" (arXiv 2510.09416, 2025) reports that permuting timestamps among
training edges often barely dents performance -- which would mean much of what
TGN and TGAT are credited with is a recency heuristic wearing a clock.

A single shuffled-versus-chronological number cannot settle that, because it
conflates two very different failures: a model that ignores order entirely, and
a model that only needs order at a coarse scale. This sweeps the shuffle window
instead and reports a curve:

    I(w) = AUC(chronological) - AUC(shuffled within windows of width w)

Timestamps are permuted only among edges falling in the same window of width w,
so every edge keeps its position in time to within w and the graph keeps its
exact edge set. Small w destroys fine ordering and preserves coarse; large w
destroys everything. Where I(w) rises off zero is the timescale at which the
domain's order actually matters.

The endpoints are the familiar experiment: w=0 is chronological, w=infinity is
the full shuffle that the paper above reports on. The middle is the part that
distinguishes "uses time" from "uses time at the resolution I claim".

This costs one full retrain per point per dataset, which is why it is not
already standard. Reproduce: `python benchmarks/order_spectrum.py`.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402

DAY = 86400.0
WINDOWS = [("chronological", 0.0), ("1 hour", 3600.0), ("1 day", DAY),
           ("1 week", 7 * DAY), ("1 month", 30 * DAY), ("1 year", 365 * DAY),
           ("full shuffle", float("inf"))]


def windowed_shuffle(ts, width, rng):
    """Permute timestamps only among edges sharing a window of the given width.

    Each edge keeps its place in time to within `width`; the edge set and the
    multiset of timestamps are both untouched. Only the pairing changes, so a
    difference in score is attributable to ordering at that scale and nothing
    else.
    """
    if width == 0.0:
        return ts.copy()
    out = ts.copy()
    bucket = (np.zeros(len(ts), dtype=np.int64) if np.isinf(width)
              else (ts.astype(np.float64) // width).astype(np.int64))
    order = np.argsort(bucket, kind="stable")
    sorted_bucket = bucket[order]
    starts = np.flatnonzero(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(order)]):
        group = order[start:end]
        if len(group) > 1:
            out[group] = ts[group][rng.permutation(len(group))]
    return out


def load_supplychain():
    scope = {"__file__": os.path.join(ROOT, "benchmarks/supplychain_link_prediction.py")}
    text = open(scope["__file__"]).read()
    exec(text.split("def artifact_dates")[0], scope)
    import pandas as pd
    chain, vertices, _ = scope["load_edges"](2003)
    return (chain["cu"].to_numpy(np.uint32), chain["su"].to_numpy(np.uint32),
            chain["ts"].to_numpy(np.uint32), vertices, "supply chain")


def load_ethereum():
    """Cached Ethereum shards, same preparation as the link-prediction study."""
    scope = {"__file__": os.path.join(ROOT, "benchmarks/ethereum_link_prediction.py")}
    exec(open(scope["__file__"]).read().split("def ranking_metrics")[0], scope)
    from ingestion.ethereum import load_transactions
    tx_glob, block_glob = scope["cached_globs"]()
    edges, _ = load_transactions(tx_glob, block_glob, quiet=True)
    return (edges["src"].to_numpy(np.uint32), edges["dst"].to_numpy(np.uint32),
            edges["ts"].to_numpy(np.uint32), int(max(edges["src"].max(),
                                                     edges["dst"].max())) + 1,
            "ethereum")


def load_jodie(name):
    scope = {"__file__": os.path.join(ROOT, "benchmarks/jodie_benchmark.py")}
    exec(open(scope["__file__"]).read().split("def main()")[0], scope)
    src, dst, ts_raw, vertices = scope["load"](name)
    return (src.astype(np.uint32), dst.astype(np.uint32),
            np.clip(ts_raw, 0, None).astype(np.uint32), vertices, name)


def run_point(src, dst, ts, vertices, width, args, rng_seed=0):
    """Train from scratch on a stream shuffled at this window, return test AUC."""
    edges = len(src)
    cut_train, cut_val = int(edges * 0.70), int(edges * 0.85)

    shuffled = ts.copy()
    shuffled[:cut_train] = windowed_shuffle(
        ts[:cut_train], width, np.random.default_rng(7))
    order = np.argsort(shuffled[:cut_train], kind="stable")
    s_g, d_g, t_g = src.copy(), dst.copy(), shuffled
    s_g[:cut_train] = src[:cut_train][order]
    d_g[:cut_train] = dst[:cut_train][order]
    t_g[:cut_train] = shuffled[:cut_train][order]

    graph = graph_engine.PCSRGraph(vertices, int(edges * 2.5),
                                   int(edges * 2.5) * 8 * 4 + (1 << 26))
    graph.insert_edges(s_g, d_g, t_g, np.zeros(edges, dtype=np.uint16))
    sampler = PCSRTemporalSampler(graph)

    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)
    model = TGATLinkModel(vertices, sampler, node_dim=args.node_dim,
                          time_dim=args.node_dim, num_layers=args.layers,
                          num_neighbors=args.neighbors)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = max(1, cut_train // args.train_events)
    tr_s, tr_d, tr_t = (s_g[:cut_train:step], d_g[:cut_train:step],
                        t_g[:cut_train:step])

    model.train()
    for _ in range(args.epochs):
        for off in range(0, len(tr_s), args.batch):
            end = min(off + args.batch, len(tr_s))
            if end - off < 8:
                continue
            optimizer.zero_grad()
            loss, _, _ = model.loss(tr_s[off:end], tr_d[off:end], tr_t[off:end],
                                    rng.integers(0, vertices, end - off))
            loss.backward()
            optimizer.step()

    # Test edges keep their real timestamps: only the history is ablated.
    model.eval()
    test = np.arange(cut_val, edges)
    if len(test) > args.max_eval:
        test = test[np.linspace(0, len(test) - 1, args.max_eval).astype(int)]
    neg = rng.integers(0, vertices, len(test))
    with torch.no_grad():
        pos = model.score(src[test], dst[test], ts[test]).numpy()
        ng = model.score(src[test], neg, ts[test]).numpy()
    return roc_auc(pos, ng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="supplychain",
                   choices=["supplychain", "ethereum", "wikipedia", "reddit"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--node-dim", type=int, default=64)
    p.add_argument("--train-events", type=int, default=30000)
    p.add_argument("--max-eval", type=int, default=4000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seeds", type=int, default=1,
                   help="repeat each point with independent seeds for error bars")
    p.add_argument("--endpoints-only", action="store_true",
                   help="chronological vs full shuffle only, the headline claim")
    args = p.parse_args()

    if args.dataset == "supplychain":
        src, dst, ts, vertices, label = load_supplychain()
    elif args.dataset == "ethereum":
        src, dst, ts, vertices, label = load_ethereum()
    else:
        src, dst, ts, vertices, label = load_jodie(args.dataset)
    span = (ts.max() - ts.min()) / DAY
    print(f"{label}: {len(src):,} edges, {vertices:,} nodes, {span:,.0f} day span")
    print(f"{args.epochs} epochs x {args.train_events:,} events per point, "
          f"{len(WINDOWS)} points\n")

    windows = ([("chronological", 0.0), ("full shuffle", float("inf"))]
               if args.endpoints_only else WINDOWS)
    spread = "     sd" if args.seeds > 1 else ""
    print(f"{'shuffle window':<18}{'AUC':>9}{spread:>8}{'I(w)':>9}{'minutes':>10}")
    print("-" * (46 + len(spread)))
    base = None
    for name, width in windows:
        if not np.isinf(width) and width > 0 and width > span * DAY:
            print(f"{name:<18}{'--':>9}{'--':>9}{'skipped (> span)':>18}")
            continue
        started = time.perf_counter()
        runs = [run_point(src, dst, ts, vertices, width, args, rng_seed=k)
                for k in range(args.seeds)]
        auc = float(np.mean(runs))
        if base is None:
            base = auc
            base_runs = runs
        sd = (f"{np.std(runs, ddof=1):>8.4f}" if args.seeds > 1 else "")
        print(f"{name:<18}{auc:>9.4f}{sd}{base - auc:>+9.4f}"
              f"{(time.perf_counter()-started)/60:>10.1f}", flush=True)
        if args.seeds > 1 and width != 0.0:
            pooled = np.sqrt(np.var(base_runs, ddof=1) / args.seeds
                             + np.var(runs, ddof=1) / args.seeds)
            if pooled > 0:
                print(f"{'':<18}{'':>9}{'':>8}  t = {(base-auc)/pooled:+.2f} "
                      f"over {args.seeds} seeds")

    print("\nI(w) is how much AUC the chronological model holds over one whose")
    print("history was scrambled within windows of width w. I(w) near zero means")
    print("order at that scale carries nothing the model was using.")


if __name__ == "__main__":
    main()
