"""
Cross-sectional trading backtest.

Takes a panel of per-country signals and a panel of prices, and reports what a
dollar-neutral book following those signals would have done -- after costs, and
with an explicit execution lag.

The timing convention is the part worth reading carefully, because it is where
backtests usually lie to their authors. See `run_backtest`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestConfig:
    """
    Args:
        direction: Sign applied to the signal. The default -1 encodes the
            hypothesis under test -- that a country facing rising expected
            conflict underperforms its peers. It is a parameter rather than a
            hard-coded minus sign so the opposite hypothesis is one flag away
            and nobody has to trust the sign by reading the source.
        execution_lag_days: Trading days between the signal date and the start
            of the return it earns. 1 means a signal computed from everything
            known by the close of day d is executed at the close of d+1 and
            earns d+1 -> d+2. GDELT publishes with a lag and closes are not
            executable in retrospect, so 0 would be dishonest.
        cost_bps: Round-trip cost charged on turnover, in basis points.
        max_weight: Per-name cap as a fraction of gross exposure, so a single
            outlier signal cannot become the whole book.
        gross_exposure: Sum of absolute weights each day.
        min_names: Days with fewer tradable names than this are skipped rather
            than traded at concentrated weights.
    """
    direction: int = -1
    execution_lag_days: int = 1
    cost_bps: float = 5.0
    max_weight: float = 0.15
    gross_exposure: float = 1.0
    min_names: int = 5
    metadata: dict = field(default_factory=dict)


def _to_wide_prices(prices):
    """(date, ticker, close) rows -> date x ticker frame."""
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    wide = frame.pivot_table(index="date", columns="ticker", values="close",
                             aggfunc="last").sort_index()
    return wide


def _to_wide_signals(signals):
    """(date, country, signal) rows -> date x country frame."""
    frame = signals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.pivot_table(index="date", columns="country", values="signal",
                             aggfunc="last").sort_index()


def build_weights(signal_frame, config):
    """
    Turn a date x country signal panel into a dollar-neutral weight panel.

    Standardisation is strictly cross-sectional: each day's signals are
    demeaned and scaled by that same day's dispersion. Using a full-sample mean
    and standard deviation is the classic silent lookahead in a panel backtest
    -- it leaks the future distribution into every past day.
    """
    centred = signal_frame.sub(signal_frame.mean(axis=1), axis=0)
    dispersion = signal_frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    z = centred.div(dispersion, axis=0)

    weights = config.direction * z
    weights = weights.clip(-3.0, 3.0)

    # Normalise so gross exposure is constant, then cap per name and
    # renormalise. Capping changes the gross, so the order matters.
    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(gross, axis=0) * config.gross_exposure
    weights = weights.clip(-config.max_weight, config.max_weight)

    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    weights = weights.div(gross, axis=0) * config.gross_exposure

    # Re-neutralise: capping can leave a small net long or short.
    weights = weights.sub(weights.mean(axis=1), axis=0)

    tradable = signal_frame.notna().sum(axis=1)
    weights[tradable < config.min_names] = np.nan
    return weights


def run_backtest(signals, prices, universe, config=None):
    """
    Args:
        signals: DataFrame with columns [date, country, signal].
        prices: DataFrame with columns [date, ticker, close].
        universe: strategy.Universe mapping country -> ticker.
        config: BacktestConfig.

    Returns:
        dict with 'daily' (per-day frame), 'weights', and 'metrics'.

    Timing, spelled out because everything depends on it:

        signal_frame.loc[d]  uses only events with ts <= close of day d
        weights.loc[d]       derived from that signal, same day
        held over            d+lag  ->  d+lag+1
        return earned        close(d+lag+1) / close(d+lag) - 1

    So the weight row is shifted forward by (lag + 1) before being multiplied
    into the return row: one shift to get from signal date to execution date,
    and one because a return indexed at t is the move from t-1 to t.
    """
    config = config or BacktestConfig()

    signal_frame = _to_wide_signals(signals)
    price_frame = _to_wide_prices(prices)

    # Map countries onto instruments and drop anything untradable.
    columns = {country: universe.ticker(country) for country in signal_frame.columns}
    columns = {country: ticker for country, ticker in columns.items()
               if ticker is not None and ticker in price_frame.columns}
    if not columns:
        raise ValueError("no signal country maps to a ticker present in the price data")

    signal_frame = signal_frame[list(columns)].rename(columns=columns)

    # Trade only on days the market was open, and only on names with a price.
    calendar = price_frame.index
    signal_frame = signal_frame.reindex(calendar)
    price_frame = price_frame[signal_frame.columns]
    signal_frame = signal_frame.where(price_frame.notna())

    weights = build_weights(signal_frame, config)

    returns = price_frame.pct_change(fill_method=None)

    held = weights.shift(config.execution_lag_days + 1)
    gross_pnl = (held * returns).sum(axis=1, min_count=1)

    # Turnover is the change in the book actually held, so it must be measured
    # on the shifted weights, not on the signal-date weights.
    turnover = held.fillna(0.0).diff().abs().sum(axis=1)
    costs = turnover * (config.cost_bps / 10_000.0)
    net_pnl = gross_pnl - costs

    daily = pd.DataFrame({
        "gross_return": gross_pnl,
        "cost": costs,
        "net_return": net_pnl,
        "turnover": turnover,
        "names": held.notna().sum(axis=1),
    })
    daily = daily[daily["names"] > 0]

    return {"daily": daily, "weights": weights, "config": config,
            "metrics": summarise(daily)}


def summarise(daily):
    """Headline statistics for a daily return series."""
    net = daily["net_return"].dropna()
    if net.empty:
        return {"days": 0}

    equity = (1.0 + net).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    years = len(net) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1.0)
    volatility = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mean_daily = float(net.mean())

    # t-stat on the mean daily return: with a few hundred observations a
    # Sharpe near 1 is routinely indistinguishable from zero, and reporting
    # Sharpe without it invites reading noise as skill.
    t_stat = (mean_daily / net.std(ddof=1) * np.sqrt(len(net))
              if net.std(ddof=1) > 0 else float("nan"))

    return {
        "days": int(len(net)),
        "total_return": total,
        "annual_return": float((1.0 + total) ** (1 / years) - 1.0) if years > 0 else float("nan"),
        "annual_volatility": volatility,
        "sharpe": float(mean_daily / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
                  if net.std(ddof=1) > 0 else float("nan"),
        "t_stat": float(t_stat),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((net > 0).mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "total_cost": float(daily["cost"].sum()),
    }


def format_metrics(name, metrics):
    if not metrics.get("days"):
        return f"{name:<22} (no tradable days)"
    return (f"{name:<22} {metrics['days']:>5}  "
            f"{metrics['annual_return']:>8.2%}  "
            f"{metrics['annual_volatility']:>7.2%}  "
            f"{metrics['sharpe']:>6.2f}  "
            f"{metrics['t_stat']:>6.2f}  "
            f"{metrics['max_drawdown']:>8.2%}  "
            f"{metrics['hit_rate']:>6.1%}  "
            f"{metrics['avg_turnover']:>7.3f}")


METRICS_HEADER = (f"{'strategy':<22} {'days':>5}  {'ann ret':>8}  {'ann vol':>7}  "
                  f"{'sharpe':>6}  {'t':>6}  {'max dd':>8}  {'hit':>6}  {'turn':>7}")
