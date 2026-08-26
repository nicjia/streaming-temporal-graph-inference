"""
Does a customer demand shock reach the supplier's supplier?

One-hop customer-to-supplier lead-lag is established in the literature and is
used here as a positive control: if the setup cannot reproduce it, nothing else
it reports means anything. The new question is the second hop -- whether the
shock continues past the direct supplier to that supplier's own suppliers.

That question needs density Compustat segment data does not have. Compustat
carries only customers above a 10% revenue disclosure threshold, giving a
two-hop degree of 1.7, which physically cannot support a second hop. FactSet
Revere, restricted to CRSP-linked firms, runs at median degree 5 and mean 11,
which can.

Neighbourhoods are resolved causally against relationship validity windows: a
2019 shock sees the supply chain as it stood in 2019, not as it stands today.

Usage:  python benchmarks/supplychain_two_hop.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from backtest.propagation import abnormal_returns, cumulative_response  # noqa: E402


def load_graph():
    """Revere supplier->customer edges, mapped to permno, with validity windows."""
    chain = pd.read_parquet(os.path.join(ROOT, "data/revere_supply_chain.parquet"))
    link = pd.read_parquet(os.path.join(ROOT, "data/revere_permno.parquet"))
    mapping = dict(zip(link["company_id"].astype(str), link["permno"]))

    chain["s"] = chain["supplier_id"].astype(str).map(mapping)
    chain["c"] = chain["customer_id"].astype(str).map(mapping)
    chain = chain.dropna(subset=["s", "c"])
    chain["s"] = chain["s"].astype(int)
    chain["c"] = chain["c"].astype(int)
    chain["start_"] = pd.to_datetime(chain["start_"], errors="coerce")
    chain["end_"] = pd.to_datetime(chain["end_"], errors="coerce")
    return chain.dropna(subset=["start_"])[["s", "c", "start_", "end_"]]


def load_returns():
    frames = [pd.read_parquet(os.path.join(ROOT, "data/crsp_dsf.parquet"),
                              columns=["permno", "date", "ret"])]
    extra = os.path.join(ROOT, "data/crsp_dsf_extra.parquet")
    if os.path.exists(extra):
        frames.append(pd.read_parquet(extra, columns=["permno", "date", "ret"]))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(["permno", "date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce").astype("float64")
    frame = frame.dropna(subset=["ret"]).sort_values("date")
    frame["ticker"] = frame["permno"].astype(int).astype(str)
    frame["close"] = frame.groupby("ticker")["ret"].transform(lambda s: (1 + s).cumprod())
    return frame[["date", "ticker", "close"]], frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shock-quantile", type=float, default=0.995)
    parser.add_argument("--max-events", type=int, default=2500)
    parser.add_argument("--max-peers", type=int, default=40)
    args = parser.parse_args()

    chain = load_graph()
    prices, raw = load_returns()
    priced = set(prices["ticker"].unique())
    print(f"supply chain: {len(chain):,} linked relationships, "
          f"{chain['s'].nunique():,} suppliers, {chain['c'].nunique():,} customers")
    print(f"returns: {len(prices):,} rows, {len(priced):,} firms")

    # Events: extreme daily moves at a customer firm.
    daily = raw.copy()
    cutoff = daily["close"].notna()
    ret = raw.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    ret = ret.astype("float64").pct_change(fill_method=None)
    relative = ret.sub(ret.mean(axis=1), axis=0)

    flat = relative.stack().reset_index()
    flat.columns = ["date", "ticker", "r"]
    threshold = flat["r"].abs().quantile(args.shock_quantile)
    customers = set(chain["c"].astype(str))
    events = flat[(flat["r"].abs() >= threshold) & (flat["ticker"].isin(customers))].copy()
    events["sign"] = np.sign(events["r"])
    if len(events) > args.max_events:
        events = events.iloc[np.linspace(0, len(events) - 1, args.max_events).astype(int)]
    print(f"shocks: {len(events):,} customer-days beyond "
          f"{threshold:.1%} relative move ({events['ticker'].nunique():,} firms)")

    # Causal adjacency as of a date, honouring relationship validity windows.
    chain_sorted = chain.sort_values("start_")
    starts = chain_sorted["start_"].to_numpy()
    ends = chain_sorted["end_"].fillna(pd.Timestamp("2100-01-01")).to_numpy()
    cust = chain_sorted["c"].to_numpy()
    supp = chain_sorted["s"].to_numpy()

    cache = {}

    def suppliers_of(firms, when):
        key = (tuple(sorted(firms)), when)
        if key in cache:
            return cache[key]
        live = (starts <= when) & (ends > when)
        target = np.isin(cust, list(firms))
        out = set(supp[live & target].tolist())
        cache[key] = out
        return out

    hop1_rows, hop2_rows, placebo_rows = [], [], []
    universe = sorted({int(x) for x in priced if x.isdigit()})
    rng = np.random.default_rng(0)

    for date, ticker, sign in zip(events["date"], events["ticker"], events["sign"]):
        when = np.datetime64(date)
        first = suppliers_of({int(ticker)}, when)
        first = {f for f in first if str(f) in priced and str(f) != ticker}
        if not first:
            continue
        second = suppliers_of(first, when) - first - {int(ticker)}
        second = {f for f in second if str(f) in priced}

        first = list(first)[:args.max_peers]
        second = list(second)[:args.max_peers]
        for f in first:
            hop1_rows.append((date, str(f), sign))
        for f in second:
            hop2_rows.append((date, str(f), sign))
        pool = rng.choice(universe, size=min(len(first) + len(second), 200), replace=False)
        for f in pool:
            if str(f) not in priced or f in first or f in second:
                continue
            placebo_rows.append((date, str(f), sign))

    abnormal = abnormal_returns(prices)
    horizons = (1, 2, 3, 5, 10, 20)
    print(f"\nobservations: hop-1 {len(hop1_rows):,} | hop-2 {len(hop2_rows):,} "
          f"| placebo {len(placebo_rows):,}")

    curves = {}
    for label, rows in (("direct suppliers (hop 1)", hop1_rows),
                        ("suppliers of suppliers (hop 2)", hop2_rows),
                        ("placebo", placebo_rows)):
        if not rows:
            continue
        d, t, s = zip(*rows)
        curves[label] = cumulative_response(abnormal, list(d), list(t), list(s), horizons)

    header = f"{'horizon':<10}" + "".join(f"{k[:26]:>28}" for k in curves)
    print("\n" + header)
    print(f"{'(days)':<10}" + "".join(f"{'CAR / t (date-clustered)':>28}" for _ in curves))
    print("-" * len(header))
    for h in horizons:
        cells = []
        for label in curves:
            mean, t_stat = curves[label][h][0], curves[label][h][1]
            cells.append(f"{mean:+.5f} / {t_stat:+6.2f}")
        print(f"{h:<10}" + "".join(f"{c:>28}" for c in cells))

    for label in curves:
        if "placebo" in label:
            continue
        print(f"\n  {label} minus placebo:")
        for h in horizons:
            a, b = curves[label][h], curves["placebo"][h]
            se = np.sqrt(a[3] ** 2 + b[3] ** 2)
            delta = a[0] - b[0]
            print(f"    h={h:<3} {delta:+.5f}  t {delta/se if se > 0 else float('nan'):+6.2f}")


if __name__ == "__main__":
    main()
