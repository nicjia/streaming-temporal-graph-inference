"""
Does insider activity at one firm move a firm that shares a director?

Board interlocks are the bridge: firm -> director -> firm. The path is
irreducibly two hops, the seats start and stop over time, and filings arrive on
no schedule, which is the case a temporal graph is for and a fixed adjacency
matrix is not.

Events are net insider buying or selling, dated by *filing* date rather than
trade date, because only the filing is public information. Peers are resolved
causally: the engine returns firms that shared a director strictly before the
event, never the interlock as it stands today.

Usage:  python benchmarks/interlock_propagation.py
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


def build_events(meta, tickers, min_value=100_000, top_quantile=0.9):
    """
    Net insider flow per firm-day, keeping only the extremes.

    Routine option exercises and small sales are noise; the informative cases
    are concentrated purchases and unusually large disposals. `sign` orients the
    response so buys and sells reinforce rather than cancel.
    """
    frame = meta[meta["ticker"].isin(tickers)].copy()
    frame = frame[frame["value"].notna() & (frame["value"] > 0)]
    frame["date"] = pd.to_datetime(frame["ts"], unit="s").dt.normalize()
    frame["signed"] = np.where(frame["code"] == "P", frame["value"], -frame["value"])

    daily = frame.groupby(["date", "ticker", "issuer"], as_index=False)["signed"].sum()
    daily = daily[daily["signed"].abs() >= min_value]

    cut = daily["signed"].abs().quantile(top_quantile)
    events = daily[daily["signed"].abs() >= cut].copy()
    events["sign"] = np.sign(events["signed"])
    return events[["date", "ticker", "issuer", "sign"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fanout", type=int, default=32)
    parser.add_argument("--max-events", type=int, default=4000)
    parser.add_argument("--top-quantile", type=float, default=0.9)
    parser.add_argument("--buys-only", action="store_true",
                        help="Purchases only. The literature finds opportunistic "
                             "buys informative and routine sells not, so this is "
                             "the sharper test.")
    args = parser.parse_args()

    edges = pd.read_parquet(os.path.join(ROOT, "data/form345/interlock_edges.parquet"))
    meta = pd.read_parquet(os.path.join(ROOT, "data/form345/meta.parquet"))
    prices = pd.read_parquet(os.path.join(ROOT, "data/form345/prices_yf.parquet"))
    cik_ticker = pd.read_parquet(os.path.join(ROOT, "data/form345/cik_ticker.parquet"))["ticker"]

    priced = set(prices["ticker"].unique())
    events = build_events(meta, priced, top_quantile=args.top_quantile)
    if len(events) > args.max_events:
        events = events.iloc[np.linspace(0, len(events) - 1, args.max_events).astype(int)]
    if args.buys_only:
        events = events[events["sign"] > 0].reset_index(drop=True)
    print(f"events: {len(events):,} firm-days ({(events['sign'] > 0).mean():.0%} buys), "
          f"{events['ticker'].nunique()} firms")

    neighborhood = BipartiteNeighborhood(edges, fanout=args.fanout)
    cik_of = dict(zip(events["ticker"], events["issuer"]))

    def neighbors(ticker, date):
        issuer = cik_of.get(ticker)
        if issuer is None:
            return []
        peers = neighborhood.peers(issuer, int(pd.Timestamp(date).timestamp()))
        out = []
        for peer in peers:
            symbol = cik_ticker.get(peer)
            if isinstance(symbol, str) and symbol in priced and symbol != ticker:
                out.append(symbol)
        return out

    covered = sum(1 for t, d in zip(events["ticker"][:300], events["date"][:300])
                  if neighbors(t, d))
    print(f"events with >=1 priced interlocked peer: {covered/3:.0f}% (of first 300)")

    result = run_event_study(events, neighbors, prices,
                             horizons=(1, 2, 3, 5, 10, 20), quiet=False)

    # Lead-lag: does the peer move before, with, or after the filing?
    print("\nLead-lag profile (peer abnormal return around the filing):")
    abnormal = abnormal_returns(prices)
    rows = []
    for k in (-10, -5, -2, 0, 2, 5, 10):
        dates, tickers, signs = [], [], []
        for date, ticker, sign in zip(events["date"], events["ticker"], events["sign"]):
            shifted = pd.Timestamp(date) + pd.Timedelta(days=k)
            for peer in neighbors(ticker, date):
                dates.append(shifted); tickers.append(peer); signs.append(sign)
        if not dates:
            continue
        response = cumulative_response(abnormal, dates, tickers, signs, [3])
        mean, t_stat, n = response[3][0], response[3][1], response[3][2]
        rows.append((k, mean, t_stat, n))
    print(f"  {'offset (days)':<16}{'CAR(3d)':>12}{'t':>8}{'n':>8}")
    print("  " + "-" * 44)
    for k, mean, t_stat, n in rows:
        note = "  before filing" if k < 0 else ("  at filing" if k == 0 else "")
        print(f"  {k:<16}{mean:>+12.4f}{t_stat:>+8.2f}{n:>8}{note}")


if __name__ == "__main__":
    main()
