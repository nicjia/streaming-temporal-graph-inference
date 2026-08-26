"""
Can the TGAT predict which supplier a firm will start sourcing from next?

This exists to answer a question the rest of the project could not: does the
model generalise? Every prior positive result for the network came from a
single dataset (Ethereum). One dataset is an anecdote. This is a second domain
with completely different dynamics -- corporate procurement decisions unfolding
over 23 years rather than wallet activity over weeks -- and the same model,
the same sampler, and the same evaluation protocol are pointed at it unchanged.

The task: a supply relationship forming is a timestamped edge. Given a customer
firm and a date, rank the supplier it actually adds against degree-matched
alternatives it did not.

Why this is the hard case by construction: a supply relationship forms once, so
essentially every test edge is a NEW pair. Pair frequency and pair recency --
the baselines that dominate repeat-heavy link prediction -- have no history to
read and cannot score the task at all. Anything above chance has to come from
the structure and timing of the firm's other relationships.

Two vendor artifacts are excluded from the evaluation window but kept as graph
history: the 2003-04-03 coverage-initiation spike (16,728 relationships dated
to the day the vendor switched the feed on) and a 2022-08 re-scrape cluster.
Both are dates on which the data was collected, not dates on which supply
chains changed, and scoring against them would measure vendor operations.

Usage:  python benchmarks/supplychain_link_prediction.py
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402


def ranking_metrics(positive, negatives):
    """MRR / Recall@k / AUC with mid-rank ties.

    Ties are scored as a coin flip, not a win. A baseline that cannot score the
    case at all returns zero for the positive and zero for every negative, and
    a strict `>` reads that as the positive winning outright. On this task that
    failure mode is not hypothetical -- pair frequency is structurally zero on
    every new pair, which is nearly the whole test set.
    """
    positive = np.asarray(positive, dtype=np.float64)
    negatives = np.asarray(negatives, dtype=np.float64)
    beaten = (negatives < positive[:, None]).sum(axis=1)
    tied = (negatives == positive[:, None]).sum(axis=1)
    rank = negatives.shape[1] - beaten - tied / 2.0 + 1.0
    return {
        "mrr": float((1.0 / rank).mean()),
        "recall@1": float((rank <= 1.5).mean()),
        "recall@10": float((rank <= 10).mean()),
        "auc": float(roc_auc(positive, negatives.ravel())),
    }


def load_edges(min_year):
    """Revere formations as a timestamped edge stream: customer -> supplier."""
    src_text = open(os.path.join(ROOT, "benchmarks/supplychain_two_hop.py")).read()
    scope = {"__file__": os.path.join(ROOT, "benchmarks/supplychain_two_hop.py")}
    exec(src_text.split("def main()")[0], scope)
    chain = scope["load_graph"]()
    chain = chain[chain["start_"].dt.year >= min_year].copy()

    # Dense vertex ids over firms that appear at all.
    firms = pd.unique(pd.concat([chain["c"], chain["s"]]))
    index = {int(f): i for i, f in enumerate(firms)}
    chain["cu"] = chain["c"].map(index).astype(np.uint32)
    chain["su"] = chain["s"].map(index).astype(np.uint32)

    # One edge per pair: the first time the relationship is recorded.
    chain = chain.sort_values("start_").drop_duplicates(["cu", "su"], keep="first")
    ts = (chain["start_"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    chain["ts"] = ts.astype(np.uint32)
    return chain.sort_values("ts").reset_index(drop=True), len(firms), index


def artifact_dates(chain, multiple):
    """Dates carrying far more formations than their neighbours -- vendor events."""
    per_day = chain.groupby(chain["start_"].dt.date).size()
    typical = per_day.median()
    return set(per_day[per_day > typical * multiple].index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-year", type=int, default=2003)
    p.add_argument("--artifact-multiple", type=float, default=8.0)
    p.add_argument("--negatives", type=int, default=100)
    p.add_argument("--max-test", type=int, default=3000)
    p.add_argument("--train-events", type=int, default=40000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--node-dim", type=int, default=64)
    p.add_argument("--checkpoint", default="supplychain_tgat.pt")
    p.add_argument("--shuffle-times", action="store_true",
                   help="permute training timestamps among training edges: same "
                        "static graph, same timestamp multiset, ordering destroyed")
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()

    chain, vertices, _ = load_edges(args.min_year)
    cu = chain["cu"].to_numpy(np.uint32)
    su = chain["su"].to_numpy(np.uint32)
    ts = chain["ts"].to_numpy(np.uint32)
    rel = np.zeros(len(cu), dtype=np.uint16)
    print(f"formations: {len(cu):,} unique pairs over {vertices:,} firms, "
          f"{chain['start_'].min().date()} -> {chain['start_'].max().date()}")

    bad = artifact_dates(chain, args.artifact_multiple)
    is_artifact = chain["start_"].dt.date.isin(bad).to_numpy()
    print(f"vendor-artifact dates excluded from evaluation: {len(bad)} dates, "
          f"{is_artifact.sum():,} formations ({is_artifact.mean():.1%})")

    cut_train = int(len(cu) * 0.75)
    cut_val = int(len(cu) * 0.85)
    split_date = chain["start_"].iloc[cut_val].date()
    print(f"chronological split: train {cut_train:,} | val {cut_val-cut_train:,} | "
          f"test {len(cu)-cut_val:,}  (test begins {split_date})")

    # The ablation that decides whether this model earns its name. Permuting
    # timestamps among training edges leaves the static graph identical edge for
    # edge, and leaves the multiset of timestamps identical value for value; the
    # only thing destroyed is which time attaches to which edge. Everything a
    # static method could read survives untouched, so any drop measured here is
    # attributable to temporal ordering and to nothing else.
    #
    # The stream is re-sorted afterwards because the PMA stores each vertex's
    # edges in time order and the sampler validates it. Re-sorting changes the
    # order edges arrive in, not which timestamp each one now carries.
    cu_g, su_g, ts_g = cu.copy(), su.copy(), ts.copy()
    if args.shuffle_times:
        perm = np.random.default_rng(7).permutation(cut_train)
        reassigned = ts[:cut_train][perm]
        order = np.argsort(reassigned, kind="stable")
        cu_g[:cut_train] = cu[:cut_train][order]
        su_g[:cut_train] = su[:cut_train][order]
        ts_g[:cut_train] = reassigned[order]
        moved = (ts_g[:cut_train] != ts[:cut_train]).mean()
        print(f"ABLATION: training timestamps permuted "
              f"({moved:.1%} of edges carry a different time; "
              f"static graph and timestamp multiset both unchanged)")

    graph = graph_engine.PCSRGraph(vertices, int(len(cu) * 2.5),
                                   int(len(cu) * 2.5) * 8 * 4 + (1 << 26))
    graph.insert_edges(cu_g, su_g, ts_g, rel)
    sampler = PCSRTemporalSampler(graph)
    print(f"graph: {graph.num_edges:,} edges, chronological: {sampler.validate()}")

    seen_pairs = set(zip(cu[:cut_train].tolist(), su[:cut_train].tolist()))
    test = np.arange(cut_val, len(cu))
    test = test[~is_artifact[cut_val:]]
    is_new = np.fromiter((( int(cu[i]), int(su[i]) ) not in seen_pairs for i in test),
                         dtype=bool, count=len(test))
    print(f"test edges after artifact removal: {len(test):,} "
          f"({is_new.mean():.1%} new pairs)")

    in_degree = np.bincount(su[:cut_train], minlength=vertices).astype(np.float64)
    candidates = np.flatnonzero(in_degree > 0)
    prior = in_degree[candidates] / in_degree[candidates].sum()
    print(f"negative pool: {len(candidates):,} suppliers seen in training, "
          f"degree-matched")

    # ---- baselines, fitted on the training window only --------------------
    suppliers_of = defaultdict(set)
    customers_of = defaultdict(set)
    for a, b in zip(cu[:cut_train], su[:cut_train]):
        suppliers_of[int(a)].add(int(b))
        customers_of[int(b)].add(int(a))
    popularity = np.log1p(in_degree)
    last_added = {}
    for b, t in zip(su[:cut_train], ts[:cut_train]):
        last_added[int(b)] = int(t)

    def score_popularity(a, b, t):
        return popularity[b]

    # The peer set depends only on the source firm, so it is computed once per
    # firm rather than once per candidate -- otherwise every test edge rebuilds
    # the same two-hop closure 101 times, which dominates the whole benchmark.
    peer_cache = {}

    def peers(a):
        a = int(a)
        hit = peer_cache.get(a)
        if hit is None:
            hit = {c for s in suppliers_of.get(a, ()) for c in customers_of.get(s, ())}
            peer_cache[a] = hit
        return hit

    def score_adamic_adar(a, b, t):
        # Firms that share customers -- the standard structural link predictor.
        common = customers_of.get(int(b), set()) & peers(a)
        return float(sum(1.0 / np.log(2.0 + len(suppliers_of.get(c, ()))) for c in common))

    def score_two_hop(a, b, t):
        # Does this supplier already serve someone my suppliers also serve?
        return float(len(peers(a) & customers_of.get(int(b), set())))

    def score_supplier_recency(a, b, t):
        seen = last_added.get(int(b))
        return 0.0 if seen is None else 1.0 / (1.0 + (int(t) - seen) / 86400.0)

    control = np.random.default_rng(1)
    baselines = {
        "control: random": lambda a, b, t: float(control.random()),
        "supplier popularity": score_popularity,
        "supplier recency": score_supplier_recency,
        "two-hop overlap": score_two_hop,
        "adamic-adar": score_adamic_adar,
        "popularity + two-hop": lambda a, b, t: popularity[b] + 2.0 * score_two_hop(a, b, t),
    }

    rng = np.random.default_rng(0)
    idx = test
    if len(idx) > args.max_test:
        idx = idx[np.linspace(0, len(idx) - 1, args.max_test).astype(int)]
    neg = rng.choice(candidates, size=(len(idx), args.negatives), p=prior)
    s_te, d_te, t_te = cu[idx], su[idx], ts[idx]

    # ---- train ------------------------------------------------------------
    torch.manual_seed(0)
    model = TGATLinkModel(vertices, sampler, node_dim=args.node_dim,
                          time_dim=args.node_dim, num_layers=args.layers,
                          num_neighbors=args.neighbors, num_relations=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    step = max(1, cut_train // args.train_events)
    tr_s, tr_d, tr_t = (cu_g[:cut_train:step], su_g[:cut_train:step],
                        ts_g[:cut_train:step])
    print(f"\ntraining on {len(tr_s):,} sampled formations, {args.epochs} epochs")

    ckpt = args.checkpoint.replace(".pt", "_shuffled.pt") if args.shuffle_times else args.checkpoint
    if args.reuse and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print(f"  loaded {ckpt}")
        args.epochs = 0

    model.train()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        for off in range(0, len(tr_s), 256):
            batch = slice(off, min(off + 256, len(tr_s)))
            n = batch.stop - batch.start
            if n < 8:
                continue
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
        torch.save(model.state_dict(), ckpt)

    # ---- evaluate ---------------------------------------------------------
    model.eval()
    header = f"{'scorer':<26}{'MRR':>8}{'R@1':>8}{'R@10':>8}{'AUC':>8}"
    print(f"\n=== NEW supply relationships ({len(idx):,} test formations, "
          f"{args.negatives} degree-matched negatives) ===")
    print(header); print("-" * len(header))
    for name, fn in baselines.items():
        pos = np.array([fn(a, b, c) for a, b, c in zip(s_te, d_te, t_te)])
        ng = np.array([[fn(a, nb, c) for nb in row] for a, row, c in zip(s_te, neg, t_te)])
        m = ranking_metrics(pos, ng)
        print(f"{name:<26}{m['mrr']:>8.3f}{m['recall@1']:>8.3f}"
              f"{m['recall@10']:>8.3f}{m['auc']:>8.3f}")
    with torch.no_grad():
        pos = model.score(s_te, d_te, t_te).numpy()
        ng = np.stack([model.score(s_te, neg[:, k], t_te).numpy()
                       for k in range(args.negatives)], axis=1)
    m = ranking_metrics(pos, ng)
    print(f"{'TGAT':<26}{m['mrr']:>8.3f}{m['recall@1']:>8.3f}"
          f"{m['recall@10']:>8.3f}{m['auc']:>8.3f}")


if __name__ == "__main__":
    main()
