"""
Signal construction: from graph state to a per-country, per-day number.

Every signal here returns the same shape -- a long frame of
[date, country, signal] -- so the backtest engine is indifferent to which one
produced it, and the model can be compared against baselines that use no model
at all. That comparison is the only thing that establishes whether the TGAT is
contributing anything.
"""

import numpy as np
import pandas as pd
import torch

CONFLICT_CLASSES = (3, 4)


def signal_dates(start_ts, end_ts, freq="B"):
    """Business days in [start_ts, end_ts), as Timestamps."""
    start = pd.Timestamp(start_ts, unit="s").normalize()
    end = pd.Timestamp(end_ts, unit="s").normalize()
    return pd.date_range(start, end, freq=freq, inclusive="left")


def cutoff_timestamp(date, cutoff_hour=0):
    """
    The instant a signal for `date` is allowed to see up to.

    Default is midnight UTC on the signal date, so the signal uses only events
    that were published before the day began. Combined with the engine's
    execution lag this is deliberately conservative: GDELT's DATEADDED is a
    publication time, and pretending an event was actionable the moment it was
    added is how a backtest quietly acquires information it never had.
    """
    return int((pd.Timestamp(date).normalize() +
                pd.Timedelta(hours=cutoff_hour)).timestamp())


# ---------------------------------------------------------------------------
# Model signal
# ---------------------------------------------------------------------------

@torch.no_grad()
def pairwise_pressure(model, node_ids, timestamp):
    """
    Expected interaction intensity for each node as of `timestamp`.

    For every node c, the mean predicted probability over all candidate
    partners v of a fresh c -> v edge. On a graph built only from conflict
    events, that reads as "how much new conflict does the model expect this
    country to be involved in".

    Factored so the encoder runs once for N nodes rather than once per pair:
    embeddings are O(N) and dominate the cost, while the predictor MLP over all
    N^2 pairs is a single batched matmul. Scoring pairs naively would be N times
    slower for identical output.
    """
    model.eval()
    node_ids = np.asarray(node_ids, dtype=np.int64)
    count = len(node_ids)
    times = np.full(count, int(timestamp), dtype=np.int64)

    embeddings = model.encoder(node_ids, times)              # (N, D)

    source = embeddings.repeat_interleave(count, dim=0)      # (N*N, D)
    target = embeddings.repeat(count, 1)                     # (N*N, D)
    probabilities = torch.sigmoid(model.predictor(source, target)).view(count, count)

    # A country interacting with itself is not a relation the graph models.
    probabilities.fill_diagonal_(0.0)
    denominator = max(count - 1, 1)
    return (probabilities.sum(dim=1) / denominator).cpu().numpy()


def model_signal(model, sampler, countries, country_ids, dates, cutoff_hour=0,
                 refresh_each_day=False):
    """
    Per-country model signal over `dates`.

    Causality comes from the sampler, not from rebuilding the graph: it only
    reads edges with timestamp strictly below the query time, which
    tests/test_tgat.py verifies produces embeddings identical to those from a
    graph physically truncated at that time. So one graph holding the entire
    corpus can be queried at any historical instant. What still has to be
    controlled is the model *weights* -- that is what the walk-forward folds are
    for.
    """
    rows = []
    for date in dates:
        if refresh_each_day:
            sampler.refresh()
        pressure = pairwise_pressure(model, country_ids, cutoff_timestamp(date, cutoff_hour))
        for country, value in zip(countries, pressure):
            rows.append((date.date().isoformat(), country, float(value)))
    return pd.DataFrame(rows, columns=["date", "country", "signal"])


def intensity_signal(model, countries, country_ids, dates, cutoff_hour=0,
                     batch_size=512):
    """
    Per-country predicted conflict rate from a TGATIntensityModel.

    Unlike `model_signal`, this is the model's direct forecast of the quantity
    the strategy cares about rather than an aggregate of ranking scores, so the
    values are comparable across countries by construction.
    """
    node_ids, timestamps, index = [], [], []
    for date in dates:
        cutoff = cutoff_timestamp(date, cutoff_hour)
        for country, node_id in zip(countries, country_ids):
            node_ids.append(node_id)
            timestamps.append(cutoff)
            index.append((pd.Timestamp(date).date().isoformat(), country))

    node_ids = np.asarray(node_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.int64)

    predictions = []
    for start in range(0, len(node_ids), batch_size):
        stop = start + batch_size
        predictions.append(model.predict(node_ids[start:stop], timestamps[start:stop]))
    predictions = np.concatenate(predictions) if predictions else np.array([])

    frame = pd.DataFrame(index, columns=["date", "country"])
    frame["signal"] = predictions
    return frame


# ---------------------------------------------------------------------------
# Baselines. A model signal that cannot beat these is not earning its keep.
# ---------------------------------------------------------------------------

def _daily_country_weight(events, weight_column):
    """
    Sum a per-event weight into (day, country) cells, counting an event against
    both of its actors.
    """
    frame = events.copy()
    frame["day"] = pd.to_datetime(frame["ts"], unit="s").dt.normalize()

    source_side = frame[["day", "src", weight_column]].rename(columns={"src": "country"})
    target_side = frame[["day", "dst", weight_column]].rename(columns={"dst": "country"})
    stacked = pd.concat([source_side, target_side], ignore_index=True)

    return stacked.pivot_table(index="day", columns="country",
                               values=weight_column, aggfunc="sum").fillna(0.0)


def _trailing(panel, dates, window_days):
    """
    Rolling sum over the previous `window_days`, evaluated strictly before each
    date.

    The shift(1) is what makes it strictly prior. Without it the window closes
    on the signal date itself, which for a daily panel means the signal sees the
    whole day it is supposed to be predicting.
    """
    calendar = pd.date_range(panel.index.min(), max(panel.index.max(), dates.max()),
                             freq="D")
    dense = panel.reindex(calendar).fillna(0.0)
    trailing = dense.rolling(window_days, min_periods=1).sum().shift(1)
    return trailing.reindex(pd.DatetimeIndex(dates).normalize())


def _long_frame(panel, countries):
    panel = panel.reindex(columns=countries)
    frame = panel.stack(future_stack=True).reset_index()
    frame.columns = ["date", "country", "signal"]
    frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    return frame.dropna(subset=["signal"])


def goldstein_signal(events, countries, dates, window_days=5):
    """
    Realised conflict intensity: mention-weighted Goldstein severity of
    conflict events over the trailing window.

    This is the baseline the model has to beat. It uses the same events and the
    same window, with no learning at all -- so any edge the model shows over it
    is attributable to the graph structure and temporal attention rather than to
    the underlying newsflow.
    """
    conflict = events[events["quad_class"].isin(CONFLICT_CLASSES)].copy()
    if conflict.empty:
        return pd.DataFrame(columns=["date", "country", "signal"])

    # Goldstein is negative for conflict; flip it so larger means more severe.
    conflict["weight"] = (conflict["num_mentions"].fillna(1.0) *
                          conflict["goldstein"].fillna(0.0).clip(upper=0.0).abs())
    panel = _daily_country_weight(conflict, "weight")
    return _long_frame(_trailing(panel, dates, window_days), countries)


def event_count_signal(events, countries, dates, window_days=5):
    """Trailing count of conflict events involving each country."""
    conflict = events[events["quad_class"].isin(CONFLICT_CLASSES)].copy()
    if conflict.empty:
        return pd.DataFrame(columns=["date", "country", "signal"])

    conflict["weight"] = 1.0
    panel = _daily_country_weight(conflict, "weight")
    return _long_frame(_trailing(panel, dates, window_days), countries)


def random_signal(countries, dates, seed=0):
    """
    Reproducible noise.

    The control. A harness that reports an attractive Sharpe for random signals
    has a bug -- in its timing, its cost model, or its return alignment -- and
    running this is how you find that out before believing anything else.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        for country in countries:
            rows.append((pd.Timestamp(date).date().isoformat(), country,
                         float(rng.normal())))
    return pd.DataFrame(rows, columns=["date", "country", "signal"])


def reversal_signal(prices, universe, countries, dates, lookback_days=1):
    """
    One-day cross-sectional reversal, using prices only.

    The control every equity signal owes the reader. Short-term reversal is a
    well-documented effect that needs no model, no graph and no event data, so
    a geopolitical signal that cannot beat it has not demonstrated that the
    geopolitics mattered. Sign convention: yesterday's losers are expected to
    outperform, so the signal is the negated trailing return and it is traded
    long.
    """
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    wide = frame.pivot_table(index="date", columns="ticker", values="close",
                             aggfunc="last").sort_index()

    tickers = {c: universe.ticker(c) for c in countries}
    tickers = {c: t for c, t in tickers.items() if t is not None and t in wide.columns}
    if not tickers:
        return pd.DataFrame(columns=["date", "country", "signal"])

    usable = [c for c in countries if c in tickers]
    panel = wide[[tickers[c] for c in usable]]
    panel.columns = usable

    trailing = panel.pct_change(lookback_days, fill_method=None)
    signal = (-trailing).reindex(pd.DatetimeIndex(dates).normalize())

    long_frame = signal.stack(future_stack=True).reset_index()
    long_frame.columns = ["date", "country", "signal"]
    long_frame["date"] = pd.to_datetime(long_frame["date"]).dt.date.astype(str)
    return long_frame.dropna(subset=["signal"])
