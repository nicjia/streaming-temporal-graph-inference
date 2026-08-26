"""
Does on-chain adoption flow lead token prices, or follow them?

The link-prediction result says the model can forecast which contract a wallet
will touch for the first time. Whether that is worth anything financially turns
on a prior question: is first-time adoption a *precursor* to price, or a
*response* to it? If flow follows price, then forecasting flow cannot forecast
returns however good the forecast is, and no amount of model work changes that.

The test is deliberately model-free. Realised adoption is measured directly from
the chain and correlated against relative token returns at a range of leads and
lags. Two controls establish that the test has power: a random signal must show
nothing, and one-week reversal -- a documented effect in crypto -- must show up.

Usage:  python benchmarks/ethereum_flow_vs_returns.py
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.ethereum import load_transactions  # noqa: E402
from ingestion.token_prices import (adoption_panel, price_history,  # noqa: E402
                                    resolve_tokens)

CACHE = os.path.join(ROOT, "data", "ethereum_flow")


def build(top_contracts, quiet=False):
    """Resolve priced tokens, pull prices, and count first-time adopters."""
    os.makedirs(CACHE, exist_ok=True)
    token_path = os.path.join(CACHE, "tokens.parquet")
    price_path = os.path.join(CACHE, "prices.parquet")
    adopt_path = os.path.join(CACHE, "adoption.parquet")

    root = os.path.expanduser("~/.cache/huggingface/hub")
    base = glob.glob(f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet"
                     f"/snapshots/*/transactions")
    if not base:
        raise SystemExit("no cached Ethereum shards; run benchmarks/ethereum_ingest.py --download N")
    table, addresses = load_transactions(base[0] + "/*.parquet",
                                         base[0].replace("transactions", "blocks") + "/*.parquet",
                                         quiet=quiet)
    # Zero-value contract calls are the ERC-20 transfer/approve pattern, so
    # their targets are the token and protocol contracts worth pricing.
    calls = table[table["relation"] == 3]

    if os.path.exists(adopt_path) and os.path.exists(price_path):
        return pd.read_parquet(price_path), pd.read_parquet(adopt_path)

    top = calls["dst"].value_counts().head(top_contracts)
    hexes = [addresses[i].hex() for i in top.index]
    tokens = resolve_tokens(hexes, quiet=quiet)
    tokens["contract_id"] = [top.index[hexes.index(a)] for a in tokens["address"]]
    tokens.to_parquet(token_path, index=False)

    start = int(table["ts"].min())
    days = int((table["ts"].max() - table["ts"].min()) / 86400) + 3
    prices = price_history(tokens["address"].tolist(), start, days, quiet=quiet)
    prices.to_parquet(price_path, index=False)

    adoption = adoption_panel(calls, set(tokens["contract_id"]), freq="W")
    adoption["ticker"] = adoption["contract"].map(dict(zip(tokens["contract_id"],
                                                           tokens["symbol"])))
    adoption.to_parquet(adopt_path, index=False)
    return prices, adoption


def weekly_panel(prices, adoption, min_vol=0.15, max_vol=6.0):
    """Aligned weekly price and adoption panels over a tradable token set."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    wide = (prices.pivot_table(index="date", columns="ticker", values="close",
                               aggfunc="last").sort_index().resample("W").last())
    # Both panels are keyed on the ISO week itself. Indexing one by week-start
    # and the other by week-end silently produces an empty join.
    wide.index = wide.index.to_period("W")
    returns = wide.pct_change(fill_method=None)

    volatility = returns.std() * np.sqrt(52)
    keep = volatility[(volatility > min_vol) & (volatility < max_vol)].index
    wide, returns = wide[keep], returns[keep]

    adoption = adoption.copy()
    adoption["week"] = pd.to_datetime(adoption["date"]).dt.to_period("W")
    counts = (adoption.pivot_table(index="week", columns="ticker", values="adopters",
                                   aggfunc="sum").reindex(columns=keep))
    counts = counts.reindex(wide.index).fillna(0.0)

    covered = counts.sum(axis=1) > 0
    return wide[covered], returns[covered], counts[covered], len(volatility), len(keep)


def rank_ic(signal, target, min_names=15):
    """Mean cross-sectional Spearman IC and its t-statistic."""
    values = []
    for period in signal.index:
        if period not in target.index:
            continue
        x, y = signal.loc[period], target.loc[period]
        ok = x.notna() & y.notna()
        if ok.sum() >= min_names and x[ok].std() > 0 and y[ok].std() > 0:
            values.append(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])
    array = np.asarray(values)
    if len(array) < 3:
        return float("nan"), float("nan"), len(array)
    return (float(array.mean()),
            float(array.mean() / array.std(ddof=1) * np.sqrt(len(array))),
            len(array))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-contracts", type=int, default=400)
    args = parser.parse_args()

    prices, adoption = build(args.top_contracts)
    wide, returns, counts, total, kept = weekly_panel(prices, adoption)
    print(f"\npanel: {counts.shape[0]} weeks x {counts.shape[1]} tokens "
          f"({total} priced contracts, {kept} past the volatility screen)")

    level = np.log1p(counts)
    surprise = level - level.rolling(8, min_periods=3).mean().shift(1)
    relative = returns.sub(returns.mean(axis=1), axis=0)
    forward = relative.shift(-1)

    print(f"\n{'signal':<34}{'IC':>10}{'t':>8}{'n':>6}   next-week relative return")
    print("-" * 74)
    rng = np.random.default_rng(0)
    for label, signal in (
            ("adoption level (log)", level),
            ("adoption surprise vs 8w trailing", surprise),
            ("adoption growth w/w", level.diff()),
            ("control: random", pd.DataFrame(rng.normal(size=counts.shape),
                                             index=counts.index, columns=counts.columns)),
            ("control: 1-week reversal", -returns)):
        ic, t, n = rank_ic(signal, forward)
        print(f"{label:<34}{ic:>+10.4f}{t:>+8.2f}{n:>6}")

    print(f"\n{'lead/lag k (weeks)':<34}{'IC':>10}{'t':>8}{'n':>6}   adoption(t) vs return(t+k)")
    print("-" * 74)
    for k in (-4, -2, -1, 0, 1, 2, 4):
        ic, t, n = rank_ic(surprise, relative.shift(-k))
        note = "  price leads flow" if k < 0 else ("  contemporaneous" if k == 0 else "")
        print(f"{('k = ' + str(k)):<34}{ic:>+10.4f}{t:>+8.2f}{n:>6}{note}")

    print("\nA profile that peaks at k<=0 and vanishes for k>0 means adoption responds to")
    print("price rather than anticipating it, so forecasting adoption cannot forecast return.")


if __name__ == "__main__":
    main()
