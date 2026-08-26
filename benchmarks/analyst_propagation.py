"""
When an analyst revises one firm, do the other firms they cover move?

Information travels firm -> analyst -> firm. The path is two hops and cannot be
collapsed: which analyst bridges the pair, and when they last touched each side,
is the mechanism. Coverage starts and stops continuously and revisions arrive at
arbitrary timestamps, so the neighbourhood must be resolved as of the event
rather than from a fixed adjacency.

Controls, all of which have caught something earlier in this project:
  placebo peers  -- random firms not covered by that analyst, matched in count
  own-event exclusion -- a peer with its own revision that day is dropped
  difference-in-differences -- cross-sectional demeaning mechanically pushes
      non-treated names the other way, so the treated-minus-placebo gap is the
      estimate, not the treated curve

Usage:  python benchmarks/analyst_propagation.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from backtest.propagation import abnormal_returns, cumulative_response, run_event_study  # noqa: E402
from backtest.two_hop import BipartiteNeighborhood  # noqa: E402
from ingestion.ibes import coverage_edges, link_to_crsp, revision_events  # noqa: E402


def load_prices(dsf, names, mapping):
    """CRSP daily returns as [date, ticker, close], keyed by IBES ticker."""
    permno_to_ibes = dict(zip(mapping["permno"], mapping["ticker"]))
    frame = dsf[dsf["permno"].isin(permno_to_ibes)].copy()
    frame["ticker"] = frame["permno"].map(permno_to_ibes)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce").astype("float64")
    frame = frame.dropna(subset=["ret"]).sort_values("date")

    # A synthetic price index from returns: the event study wants levels, and
    # CRSP RET already handles splits, dividends and delisting.
    frame["close"] = frame.groupby("ticker")["ret"].transform(lambda s: (1 + s).cumprod())
    return frame[["date", "ticker", "close"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fanout", type=int, default=48)
    parser.add_argument("--max-events", type=int, default=3000)
    parser.add_argument("--top-quantile", type=float, default=0.95)
    parser.add_argument("--max-coverage", type=int, default=50,
                        help="Drop analyst codes covering more than this many firms; "
                             "the tail runs to 489 and is desk-level rather than a person")
    args = parser.parse_args()

    revisions = pd.read_parquet(os.path.join(ROOT, "data/ibes_revisions.parquet"))
    names = pd.read_parquet(os.path.join(ROOT, "data/crsp_names.parquet"))
    idsum = pd.read_parquet(os.path.join(ROOT, "data/ibes_idsum.parquet"))
    dsf = pd.read_parquet(os.path.join(ROOT, "data/crsp_dsf.parquet"))

    mapping = link_to_crsp(revisions, idsum, names)
    prices = load_prices(dsf, names, mapping)
    priced = set(prices["ticker"].unique())
    print(f"prices: {len(prices):,} rows, {len(priced):,} firms, "
          f"{prices['date'].min().date()} -> {prices['date'].max().date()}")

    edges = coverage_edges(revisions)
    breadth = edges.groupby("bridge")["entity"].nunique()
    edges = edges[edges["bridge"].isin(breadth[breadth <= args.max_coverage].index)]
    edges = edges[edges["entity"].isin(priced)]
    print(f"coverage after screens: {len(edges):,} edges, "
          f"{edges['entity'].nunique():,} firms, {edges['bridge'].nunique():,} analysts")

    events = revision_events(revisions, top_quantile=args.top_quantile)
    events = events[events["ticker"].isin(priced)]
    if len(events) > args.max_events:
        events = events.iloc[np.linspace(0, len(events) - 1, args.max_events).astype(int)]
    print(f"events used: {len(events):,}")

    neighborhood = BipartiteNeighborhood(edges, fanout=args.fanout)
    analyst_of = dict(zip(range(len(events)), events["analys"]))
    lookup = {}

    def neighbors(ticker, date):
        # Peers are the OTHER firms this analyst covers as of the event.
        stamp = int(pd.Timestamp(date).timestamp())
        key = (ticker, stamp // 86400)
        if key in lookup:
            return lookup[key]
        peers = [p for p in neighborhood.peers(ticker, stamp)
                 if p in priced and p != ticker]
        lookup[key] = peers
        return peers

    sample = [len(neighbors(t, d)) for t, d in zip(events["ticker"][:400], events["date"][:400])]
    print(f"peers per event: median {np.median(sample):.0f} mean {np.mean(sample):.1f} "
          f"| {np.mean(np.array(sample) > 0):.0%} have >=1")

    result = run_event_study(events, neighbors, prices,
                             horizons=(1, 2, 3, 5, 10, 20), quiet=False)

    print("\nLead-lag (peer 3-day CAR at offsets around the revision):")
    abnormal = abnormal_returns(prices)
    print(f"  {'offset (days)':<16}{'CAR(3d)':>12}{'t':>8}{'n':>8}")
    print("  " + "-" * 44)
    for k in (-10, -5, -2, 0, 2, 5, 10):
        dates, tickers, signs = [], [], []
        for date, ticker, sign in zip(events["date"], events["ticker"], events["sign"]):
            shifted = pd.Timestamp(date) + pd.Timedelta(days=k)
            for peer in neighbors(ticker, date):
                dates.append(shifted); tickers.append(peer); signs.append(sign)
        if not dates:
            continue
        cell = cumulative_response(abnormal, dates, tickers, signs, [3])[3]
        note = "  before revision" if k < 0 else ("  at revision" if k == 0 else "")
        print(f"  {k:<16}{cell[0]:>+12.4f}{cell[1]:>+8.2f}{cell[2]:>8}{note}")


if __name__ == "__main__":
    main()
