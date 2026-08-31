"""
Does *network-propagated* token adoption lead returns when raw adoption does not?

Raw first-time adoption follows price in this sample. That does not rule out a
more specific mechanism: a wallet may adopt a token after interacting with a
wallet that already uses it. The time-respecting motif is

    existing adopter(c)  --interacts with-->  new wallet  --adopts-->  token c

and it has an arrow of time. Reverse the first two events and the static graph,
degrees, wallets, token and adoption count are unchanged, but the propagation
interpretation disappears.

At the start of each week, the PCSR sampler freezes the information set. For
every wallet that first adopts a priced token during that week, we ask whether
one of its K most recent counterparties was already an adopter before the week
began. The signal is not the raw count. Within each week we permute token labels
among the same adoption events, preserving every token's count and every
wallet's activity, and subtract the expected exposed-adopter count under that
placebo. This isolates whether the observed token is unusually aligned with the
wallet's prior network.

The resulting excess-propagation score is formed at week-end and tested against
the next week's cross-sectional token return. No same-week return enters the
signal. Results include rank IC, a dollar-neutral quintile portfolio, turnover,
cost sensitivity, and a lead/lag profile.

Usage: python benchmarks/ethereum_propagated_adoption.py
"""

import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from ingestion.ethereum import load_transactions  # noqa: E402
from models import PCSRTemporalSampler  # noqa: E402

def cached_ethereum():
    root = os.path.expanduser("~/.cache/huggingface/hub")
    base = glob.glob(f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet"
                     f"/snapshots/*/transactions")
    if not base:
        raise SystemExit("no cached Ethereum shards")
    return base[0] + "/*.parquet", base[0].replace("transactions", "blocks") + "/*.parquet"


def load_weekly_prices():
    from ethereum_flow_vs_returns import weekly_panel

    prices = pd.read_parquet(os.path.join(ROOT, "data/ethereum_flow/prices.parquet"))
    adoption = pd.read_parquet(os.path.join(ROOT, "data/ethereum_flow/adoption.parquet"))
    return weekly_panel(prices, adoption)


def load_daily_prices():
    prices = pd.read_parquet(os.path.join(ROOT, "data/ethereum_flow/prices.parquet"))
    prices["date"] = pd.to_datetime(prices["date"])
    wide = (prices.pivot_table(index="date", columns="ticker", values="close",
                               aggfunc="last").sort_index().resample("D").last())
    wide.index = wide.index.to_period("D")
    returns = wide.pct_change(fill_method=None)
    volatility = returns.std() * np.sqrt(365)
    keep = volatility[(volatility > 0.15) & (volatility < 6.0)].index
    return wide[keep], returns[keep], len(volatility), len(keep)


def build_panel(freq="W", neighbors=30, placebos=10, seed=0, rebuild=False):
    cache = os.path.join(ROOT, "data", "ethereum_flow",
                         f"propagated_adoption_{freq}_k{neighbors}_p{placebos}.parquet")
    if os.path.exists(cache) and not rebuild:
        print(f"loading cached motif panel: {cache}")
        return pd.read_parquet(cache)

    tx_glob, block_glob = cached_ethereum()
    events, addresses = load_transactions(tx_glob, block_glob, quiet=True)
    tokens = pd.read_parquet(os.path.join(ROOT, "data/ethereum_flow/tokens.parquet"))
    token_ids = set(tokens["contract_id"].astype(int))
    ticker_of = dict(zip(tokens["contract_id"].astype(int), tokens["symbol"]))

    adoption = (events[(events["relation"] == 3) & events["dst"].isin(token_ids)]
                .sort_values("ts", kind="stable")
                .drop_duplicates(["src", "dst"], keep="first")
                [["src", "dst", "ts"]].copy())
    adoption["period"] = pd.to_datetime(adoption["ts"], unit="s").dt.to_period(freq)
    print(f"events {len(events):,} | addresses {len(addresses):,} | "
          f"priced-token first adoptions {len(adoption):,}")

    # Undirected contact history. Two adjacent rows share a timestamp and the
    # original event stream is chronological, so no sort or temporary 2N-row
    # frame is needed.
    n = len(events)
    bi_src = np.empty(2 * n, dtype=np.uint32)
    bi_dst = np.empty(2 * n, dtype=np.uint32)
    bi_ts = np.empty(2 * n, dtype=np.uint32)
    src = events["src"].to_numpy(np.uint32)
    dst = events["dst"].to_numpy(np.uint32)
    ts = events["ts"].to_numpy(np.uint32)
    bi_src[0::2], bi_src[1::2] = src, dst
    bi_dst[0::2], bi_dst[1::2] = dst, src
    bi_ts[0::2], bi_ts[1::2] = ts, ts
    del src, dst, ts, events

    capacity = int(len(bi_src) * 1.5)
    arena = capacity * 32 + (1 << 27)
    graph = graph_engine.PCSRGraph(len(addresses), capacity, arena)
    graph.insert_edges(bi_src, bi_dst, bi_ts,
                       np.zeros(len(bi_src), dtype=np.uint16))
    sampler = PCSRTemporalSampler(graph)
    print(f"contact graph {graph.num_edges:,} directed contacts, "
          f"chronological={sampler.validate()}")
    del bi_src, bi_dst, bi_ts

    rng = np.random.default_rng(seed)
    prior = defaultdict(set)
    rows = []
    started = time.perf_counter()

    for number, (period, part) in enumerate(adoption.groupby("period", sort=True), 1):
        # Freeze at the period boundary. An adoption later in this period cannot
        # expose another wallet until the next one, removing within-period
        # ambiguity and making the signal available at the following rebalance.
        cutoff = int(period.start_time.timestamp())
        wallets = part["src"].to_numpy(np.int64)
        labels = part["dst"].to_numpy(np.int64)
        query_times = np.full(len(part), cutoff, dtype=np.int64)
        nbr, _, mask = sampler.sample(wallets, query_times, neighbors)

        # Materialize valid neighbours once. Sets make token-conditioned tests
        # stop at the first match and avoid an enormous wallet x token matrix.
        histories = [row[m].astype(np.int64, copy=False)
                     for row, m in zip(nbr, mask)]

        def exposed_count(assigned):
            counts = defaultdict(int)
            for history, token in zip(histories, assigned):
                known = prior[int(token)]
                if known and any(int(node) in known for node in history):
                    counts[int(token)] += 1
            return counts

        actual = exposed_count(labels)
        null = defaultdict(list)
        for _ in range(placebos):
            draw = exposed_count(labels[rng.permutation(len(labels))])
            for token in np.unique(labels):
                null[int(token)].append(draw.get(int(token), 0))

        token_counts = pd.Series(labels).value_counts()
        for token, total in token_counts.items():
            draws = np.asarray(null[int(token)], dtype=np.float64)
            observed = actual.get(int(token), 0)
            expected = float(draws.mean()) if len(draws) else 0.0
            deviation = float(draws.std(ddof=1)) if len(draws) > 1 else 0.0
            z = ((observed - expected) / deviation if deviation > 0
                 else float(observed - expected))
            rows.append((period.start_time, int(token), ticker_of[int(token)],
                         int(total), observed, expected, z, len(prior[int(token)])))

        # Only after scoring the week do its adopters enter the historical set.
        for wallet, token in zip(wallets, labels):
            prior[int(token)].add(int(wallet))

        if number % (50 if freq == "D" else 10) == 0:
            unit = "days" if freq == "D" else "weeks"
            print(f"  {number:>3} {unit} | {len(rows):>6,} token-periods | "
                  f"{(time.perf_counter()-started):.0f}s", flush=True)

    panel = pd.DataFrame(rows, columns=["date", "contract", "ticker", "adopters",
                                        "exposed", "expected", "propagation_z",
                                        "prior_adopters"])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    panel.to_parquet(cache, index=False)
    return panel


def rank_ic(signal, target, min_names=15):
    values = []
    for period in signal.index.intersection(target.index):
        x, y = signal.loc[period], target.loc[period]
        valid = x.notna() & y.notna()
        if valid.sum() >= min_names and x[valid].std() > 0 and y[valid].std() > 0:
            values.append(x[valid].rank().corr(y[valid].rank()))
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return float("nan"), float("nan"), len(values)
    return (float(values.mean()),
            float(values.mean() / values.std(ddof=1) * np.sqrt(len(values))),
            len(values))


def newey_west_t(values, lags=4):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    centered = x - x.mean()
    variance = np.dot(centered, centered) / len(x)
    for lag in range(1, min(lags, len(x) - 1) + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / len(x)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    se = np.sqrt(max(variance, 0.0) / len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def portfolio(signal, forward_returns, cost_bps=0.0, min_names=15,
              periods_per_year=52, nw_lags=4):
    dates = signal.index.intersection(forward_returns.index)
    previous = pd.Series(0.0, index=signal.columns)
    gross, net, turnover = [], [], []
    for date in dates:
        score, ret = signal.loc[date], forward_returns.loc[date]
        valid = score.notna() & ret.notna()
        weight = pd.Series(0.0, index=signal.columns)
        if valid.sum() >= min_names and score[valid].nunique() >= 5:
            ranked = score[valid].rank(pct=True, method="average")
            long = ranked[ranked >= 0.8].index
            short = ranked[ranked <= 0.2].index
            if len(long) and len(short):
                weight[long] = 0.5 / len(long)
                weight[short] = -0.5 / len(short)
        turn = float((weight - previous).abs().sum())
        pnl = float((weight * ret.fillna(0.0)).sum())
        gross.append(pnl)
        net.append(pnl - turn * cost_bps / 10000.0)
        turnover.append(turn)
        previous = weight
    gross, net = np.asarray(gross), np.asarray(net)
    return {
        "weeks": len(net),
        "gross_sharpe": float(gross.mean() / gross.std(ddof=1) * np.sqrt(periods_per_year)),
        "net_sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year)),
        "nw_t": newey_west_t(net, nw_lags),
        "mean_turnover": float(np.mean(turnover)),
        "annual_return": float(net.mean() * periods_per_year),
    }


def slow_portfolio(signal, counts, forward_returns, start, end,
                   rebalance=7, top_n=50, cost_bps=25):
    """Fixed low-turnover specification selected before the final holdout.

    Eligibility uses only trailing 30-period adoption. Positions rebalance once
    every seven periods and are carried in between; no future liquidity or
    end-of-sample universe rank enters the decision.
    """
    previous = pd.Series(0.0, index=signal.columns)
    pnl, turns = [], []
    dates = signal.index[(signal.index >= start) & (signal.index <= end)]
    for number, date in enumerate(dates):
        if number % rebalance == 0:
            trailing = counts.loc[:date].tail(30).sum()
            eligible = trailing.rank(ascending=False, method="first") <= top_n
            score = signal.loc[date]
            valid = score.notna() & eligible
            weight = pd.Series(0.0, index=signal.columns)
            if valid.sum() >= 15:
                rank = score[valid].rank(pct=True, method="average")
                long = rank[rank >= 0.8].index
                short = rank[rank <= 0.2].index
                if len(long) and len(short):
                    weight[long] = 0.5 / len(long)
                    weight[short] = -0.5 / len(short)
        else:
            weight = previous.copy()
        turn = float((weight - previous).abs().sum())
        ret = float((weight * forward_returns.loc[date].fillna(0.0)).sum())
        pnl.append(ret - turn * cost_bps / 10000.0)
        turns.append(turn)
        previous = weight
    values = np.asarray(pnl, dtype=np.float64)
    return {
        "sharpe": float(values.mean() / values.std(ddof=1) * np.sqrt(365)),
        "nw_t": newey_west_t(values, 7),
        "annual_return": float(values.mean() * 365),
        "turnover": float(np.mean(turns)),
    }


def fama_macbeth_attribution(graph_signal, raw_adoption, same_day_return,
                             next_return, start):
    """Daily rank regression; reports whether graph order adds information."""
    coefficients = []
    for date in graph_signal.index[graph_signal.index >= start]:
        frame = pd.DataFrame({
            "graph": graph_signal.loc[date],
            "adoption": raw_adoption.loc[date],
            "return": same_day_return.loc[date],
            "target": next_return.loc[date],
        }).dropna()
        if len(frame) < 15:
            continue
        for column in frame:
            frame[column] = frame[column].rank(pct=True) - 0.5
        design = np.c_[np.ones(len(frame)),
                       frame[["graph", "adoption", "return"]].to_numpy()]
        coefficients.append(np.linalg.lstsq(
            design, frame["target"].to_numpy(), rcond=None)[0][1:])
    array = np.asarray(coefficients, dtype=np.float64)
    return [(float(array[:, j].mean()), newey_west_t(array[:, j], 7))
            for j in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--placebos", type=int, default=10)
    parser.add_argument("--freq", choices=["D", "W"], default="D")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    motif = build_panel(args.freq, args.neighbors, args.placebos,
                        rebuild=args.rebuild)
    if args.freq == "W":
        _, returns, raw_counts, total, kept = load_weekly_prices()
        periods_per_year, nw_lags = 52, 4
    else:
        _, returns, total, kept = load_daily_prices()
        raw_counts = None
        periods_per_year, nw_lags = 365, 7
    motif["period"] = pd.to_datetime(motif["date"]).dt.to_period(args.freq)

    def wide(column):
        return (motif.pivot_table(index="period", columns="ticker", values=column,
                                  aggfunc="sum")
                .reindex(index=returns.index, columns=returns.columns))

    z = wide("propagation_z")
    excess = wide("exposed") - wide("expected")
    counts = (raw_counts.reindex_like(returns) if raw_counts is not None
              else wide("adopters").fillna(0.0))
    relative = returns.sub(returns.mean(axis=1), axis=0)
    forward = relative.shift(-1)

    # A change, not a level: persistent token-specific graph density should not
    # become a permanent long or short. The eight-week trailing mean is shifted
    # so the current observation is never part of its own expectation.
    signals = {
        "propagation z": z,
        "propagation z surprise": z - z.rolling(8, min_periods=3).mean().shift(1),
        "excess exposed adopters": np.sign(excess) * np.log1p(excess.abs()),
        "raw adoption surprise": (np.log1p(counts)
                                   - np.log1p(counts).rolling(8, min_periods=3).mean().shift(1)),
        "control: 1-period reversal": -returns,
    }

    unit = "days" if args.freq == "D" else "weeks"
    print(f"\ntradable panel: {returns.shape[0]} {unit} x {returns.shape[1]} tokens "
          f"({total} priced, {kept} through volatility screen)")
    print(f"\n{'signal':<30}{'IC':>9}{'t':>8}{'n':>6}  next-period relative return")
    print("-" * 70)
    for name, signal in signals.items():
        ic, t, n = rank_ic(signal, forward)
        print(f"{name:<30}{ic:>+9.4f}{t:>+8.2f}{n:>6}")

    best = signals["propagation z surprise"]
    print("\nfull-sample turnover stress (diagnostic, not a holdout result)")
    print(f"{'cost (bps per unit turnover)':<31}{'gross S':>9}{'net S':>9}"
          f"{'NW t':>8}{'turn':>8}{'ann ret':>10}")
    print("-" * 78)
    for cost in (0, 25, 50, 100):
        result = portfolio(best, forward, cost, periods_per_year=periods_per_year,
                           nw_lags=nw_lags)
        print(f"{cost:<31}{result['gross_sharpe']:>9.2f}{result['net_sharpe']:>9.2f}"
              f"{result['nw_t']:>8.2f}{result['mean_turnover']:>8.2f}"
              f"{result['annual_return']:>+10.1%}")

    print(f"\n{'return offset k (periods)':<30}{'IC':>9}{'t':>8}{'n':>6}")
    print("-" * 56)
    for k in (-4, -2, -1, 0, 1, 2, 4):
        ic, t, n = rank_ic(best, relative.shift(-k))
        note = " price leads" if k < 0 else (" same period" if k == 0 else "")
        print(f"{('k = ' + str(k)):<30}{ic:>+9.4f}{t:>+8.2f}{n:>6}{note}")

    if args.freq == "D":
        # The middle 20% was used once to select weekly rebalancing and a
        # trailing-activity top-50 universe from a small predeclared grid. The
        # last 20% remained untouched until this fixed specification was run.
        available = best.dropna(how="all").index
        holdout = available[int(len(available) * 0.8)]
        end = available[-1]
        print(f"\nfixed strategy holdout: {holdout} through {end}; "
              "7-day rebalance, trailing-activity top 50, 25 bps")
        print(f"{'signal':<28}{'net S':>9}{'NW t':>9}{'ann ret':>10}{'turn':>9}")
        print("-" * 66)
        for name, signal in (("graph propagation", best),
                             ("raw adoption", signals["raw adoption surprise"]),
                             ("one-day reversal", -returns)):
            result = slow_portfolio(signal, counts, forward, holdout, end)
            print(f"{name:<28}{result['sharpe']:>9.2f}{result['nw_t']:>9.2f}"
                  f"{result['annual_return']:>+10.1%}{result['turnover']:>9.2f}")

        attribution = fama_macbeth_attribution(
            best, signals["raw adoption surprise"], relative, forward, holdout)
        print("\nincremental daily rank regression on the same holdout")
        for name, (coefficient, t_stat) in zip(
                ("graph propagation", "raw adoption", "same-day return"),
                attribution):
            print(f"  {name:<20} coefficient {coefficient:+.4f}  NW t {t_stat:+.2f}")


if __name__ == "__main__":
    main()
