"""
Does a shock propagate better along edges that multiple sources agree on?

A standard event study on the multi-source graph. On the day a firm has an
extreme move, measure its neighbours' abnormal returns over the following days,
split by how many independent sources vouch for the edge, against a
degree-matched placebo. If corroboration carries information, multiply-sourced
neighbours should move more than single-sourced ones.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from ingestion.entity_resolve import EntityResolver  # noqa: E402
from ingestion.edgar import company_tickers  # noqa: E402
import unified_graph as ug  # noqa: E402


def load_returns():
    """permno daily returns joined to ticker, wide as ticker x date."""
    dsf = pd.read_parquet("data/crsp_dsf.parquet", columns=["permno", "date", "ret"])
    names = pd.read_parquet("data/crsp_names.parquet",
                            columns=["permno", "crsp_ticker", "namedt", "nameendt"])
    # Most recent ticker per permno is adequate for a cross-sectional test.
    names = (names.sort_values("nameendt").groupby("permno").last()
             .reset_index()[["permno", "crsp_ticker"]])
    dsf = dsf.merge(names, on="permno", how="inner")
    dsf["ticker"] = dsf["crsp_ticker"].astype(str).str.upper()
    dsf["date"] = pd.to_datetime(dsf["date"])
    dsf["ret"] = pd.to_numeric(dsf["ret"], errors="coerce")
    return dsf.dropna(subset=["ret"])


def abnormal(dsf):
    """Return minus the equal-weight cross-sectional mean that day."""
    mkt = dsf.groupby("date")["ret"].transform("mean")
    dsf = dsf.assign(abn=dsf["ret"] - mkt)
    return dsf.pivot_table(index="date", columns="ticker", values="abn",
                           aggfunc="first")


def edge_support(e8k, einter, esupp):
    """firm-pair -> bitmask of sources (1 text, 2 interlock, 4 supply)."""
    support = defaultdict(int)
    for edges, bit in ((e8k, 1), (einter, 2), (esupp, 4)):
        for a, b, *_ in edges:
            support[(min(a, b), max(a, b))] |= bit
    return support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges-8k", default="data/edgar/edges/*/edges.*.jsonl")
    parser.add_argument("--interlock", default="data/form345/202[024]q*_form345.zip")
    parser.add_argument("--shock", type=float, default=0.10,
                        help="abs abnormal move that counts as a shock event")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    resolver = EntityResolver(company_tickers())
    listed = set(company_tickers()["ticker"])
    print("loading edges ...", flush=True)
    e8k = ug.load_8k(args.edges_8k, resolver)
    einter = ug.load_interlock(args.interlock, listed)
    esupp = ug.load_supply_chain(listed)
    support = edge_support(e8k, einter, esupp)

    neighbours = defaultdict(dict)  # firm -> {neighbour: n_sources}
    for (a, b), mask in support.items():
        n = bin(mask).count("1")
        neighbours[a][b] = n
        neighbours[b][a] = n

    print("loading returns ...", flush=True)
    dsf = load_returns()
    abn = abnormal(dsf).sort_index()

    # Normalise neighbour keys to bare tickers once, and keep only firms that
    # both have neighbours and have return data -- there is no point scanning
    # shocks for a firm with no edges.
    graph = {}
    for firm, nb in neighbours.items():
        t = firm[1] if isinstance(firm, tuple) else firm
        graph[t] = {(k[1] if isinstance(k, tuple) else k): v for k, v in nb.items()}

    # Everything below is numpy on a (dates x tickers) matrix. Per-event pandas
    # .loc was O(events x neighbours) of slow label lookups -- 245k events made
    # it never finish. Precomputing a forward-CAR matrix makes each event a
    # constant-time row gather.
    col_ix = {t: i for i, t in enumerate(abn.columns)}
    M = abn.to_numpy(dtype=np.float64)                 # dates x tickers
    filled = np.nan_to_num(M)
    H = args.horizon
    # car_fwd[d, j] = sum of abnormal returns for ticker j over days d+1..d+H
    csum = np.cumsum(filled, axis=0)
    car_fwd = np.full_like(M, np.nan)
    n_dates = M.shape[0]
    for d in range(n_dates - H):
        car_fwd[d] = csum[d + H] - csum[d]

    shock_firms = [t for t in graph if t in col_ix]
    buckets = defaultdict(list)
    placebo = []
    rng = np.random.default_rng(0)
    all_ix = np.array([col_ix[t] for t in abn.columns])
    n_events = 0
    for firm in shock_firms:
        j = col_ix[firm]
        col = M[:, j]
        shock_days = np.where(np.abs(col) >= args.shock)[0]
        shock_days = shock_days[shock_days < n_dates - H]
        norm = graph[firm]
        peer_ix = [(col_ix[p], n) for p, n in norm.items() if p in col_ix]
        for d in shock_days:
            sign = np.sign(col[d])
            n_events += 1
            for pj, nsrc in peer_ix:
                car = sign * car_fwd[d, pj]
                if car == car:
                    buckets[nsrc].append(car)
            # degree-matched placebo: as many random non-neighbours as there
            # are real ones, drawn once per event
            for pj in rng.choice(all_ix, size=min(len(peer_ix), 20), replace=False):
                car = sign * car_fwd[d, pj]
                if car == car:
                    placebo.append(car)
    print(f"shock events (firms with edges): {n_events:,}", flush=True)

    print(f"\n{'='*60}\nMULTI-SOURCE SHOCK PROPAGATION ({args.horizon}d CAR, "
          f"signed by shock)\n{'='*60}")
    print(f"{'edge type':<22}{'n':>8}{'mean CAR':>12}{'t':>8}")
    def stat(name, xs):
        xs = np.array(xs)
        if len(xs) < 2:
            print(f"{name:<22}{len(xs):>8}{'--':>12}")
            return
        t = xs.mean() / (xs.std(ddof=1) / np.sqrt(len(xs)))
        print(f"{name:<22}{len(xs):>8}{xs.mean():>12.4f}{t:>8.2f}")
    stat("placebo (non-neighbour)", placebo)
    for n in sorted(buckets):
        stat(f"{n} source(s)", buckets[n])

    print("\nIf corroboration carries information, mean CAR should rise with "
          "source count and exceed the placebo.")


if __name__ == "__main__":
    main()
