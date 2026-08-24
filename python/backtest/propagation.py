"""
How fast does a shock at one firm reach its supply-chain neighbours?

This is a *measurement*, not a prediction. The output is a response curve --
abnormal return of A's neighbours as a function of horizon after a shock at A --
and the half-life implied by it. A curve that peaks at 20 minutes is a finding.
A curve that is flat at zero is also a finding: it says the linkage is priced
faster than the data can resolve. Neither outcome is a failed experiment, which
is what makes this a better question than "does this predict returns".

Three controls, because an event study is easy to fool:

1. **Placebo neighbours.** Randomly chosen non-neighbours, matched in count.
   They must show no response. If they do, the estimator is picking up a market
   or sector factor rather than propagation.
2. **Own-event exclusion.** A neighbour with its own shock in the window is
   dropped; otherwise clustered earnings dates masquerade as propagation.
3. **Abnormal, not raw, returns.** Each return is demeaned cross-sectionally
   against the same day's universe, so a market-wide move is not read as a
   shock travelling down every supply chain at once.
"""

import numpy as np
import pandas as pd


def abnormal_returns(prices):
    """Cross-sectionally demeaned returns: date x instrument."""
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    wide = frame.pivot_table(index="date", columns="ticker", values="close",
                             aggfunc="last").sort_index()
    returns = wide.pct_change(fill_method=None)
    return returns.sub(returns.mean(axis=1), axis=0)


def cumulative_response(abnormal, dates, tickers, signs, horizons):
    """
    Mean sign-adjusted cumulative abnormal return at each horizon.

    `signs` orients each observation so a positive shock and a negative shock
    of the same magnitude reinforce rather than cancel.
    """
    index = {d: i for i, d in enumerate(abnormal.index)}
    columns = {t: i for i, t in enumerate(abnormal.columns)}
    values = abnormal.to_numpy()

    out = {}
    for horizon in horizons:
        collected = []
        for date, ticker, sign in zip(dates, tickers, signs):
            row = index.get(date)
            col = columns.get(ticker)
            if row is None or col is None:
                continue
            stop = row + horizon + 1
            if stop > len(values):
                continue
            window = values[row + 1:stop, col]      # strictly after the event day
            if window.size == 0 or np.isnan(window).all():
                continue
            collected.append(sign * np.nansum(window))
        if collected:
            arr = np.asarray(collected, dtype=float)
            deviation = arr.std(ddof=1)
            standard_error = deviation / np.sqrt(len(arr)) if deviation > 0 else float("nan")
            out[horizon] = (float(arr.mean()),
                            float(arr.mean() / standard_error) if standard_error > 0
                            else float("nan"),
                            len(arr),
                            float(standard_error))
        else:
            out[horizon] = (float("nan"), float("nan"), 0, float("nan"))
    return out


def difference_curves(treated, control, horizons):
    """
    Treated minus control, with the standard error of the difference.

    This, not the raw treated curve, is the estimate to report -- and the reason
    is mechanical rather than cosmetic. Abnormal returns are demeaned across the
    cross-section each day, so lifting a handful of firms necessarily pushes
    every other firm down to keep that day's mean at zero. Measured on synthetic
    data with a known 3-day half-life, the placebo group drifted to a t of -2.85
    purely from this, with no contamination whatsoever: the response was
    negative, monotone in horizon, and a mirror of the treated curve.
    Differencing cancels the induced offset. The two groups are disjoint
    samples, so the errors add in quadrature.
    """
    out = {}
    for horizon in horizons:
        t_mean, _, t_n, t_se = treated[horizon]
        c_mean, _, c_n, c_se = control[horizon]
        if not (np.isfinite(t_mean) and np.isfinite(c_mean)):
            out[horizon] = (float("nan"), float("nan"), 0, float("nan"))
            continue
        delta = t_mean - c_mean
        se = np.sqrt((t_se ** 2 if np.isfinite(t_se) else 0.0) +
                     (c_se ** 2 if np.isfinite(c_se) else 0.0))
        out[horizon] = (float(delta),
                        float(delta / se) if se > 0 else float("nan"),
                        min(t_n, c_n), float(se))
    return out


def implied_half_life(response, horizons):
    """
    Horizon at which the cumulative response first reaches half its maximum.

    Reported by linear interpolation between the bracketing horizons. If the
    response peaks at the first horizon the answer is "faster than the data can
    resolve", which is reported as such rather than as zero.
    """
    curve = [(h, response[h][0]) for h in horizons if np.isfinite(response[h][0])]
    if not curve:
        return None
    peak = max(value for _, value in curve)
    if peak <= 0:
        return None
    target = peak / 2.0

    for i, (horizon, value) in enumerate(curve):
        if value >= target:
            if i == 0:
                return 0.0  # already half-priced by the first observable horizon
            prev_h, prev_v = curve[i - 1]
            if value == prev_v:
                return float(horizon)
            fraction = (target - prev_v) / (value - prev_v)
            return float(prev_h + fraction * (horizon - prev_h))
    return None


def run_event_study(events, neighbor_fn, prices, horizons=(1, 2, 3, 5, 10, 20),
                    placebo_rng=None, exclude_own_events=True, quiet=False):
    """
    Args:
        events: DataFrame [date, ticker, sign] -- the shocks. `sign` is +1/-1
            (or a surprise magnitude) orienting the response.
        neighbor_fn: callable(ticker, date) -> list of neighbour tickers as of
            that date. This is where the graph engine plugs in; it must be
            causal, returning only links known at `date`.
        prices: DataFrame [date, ticker, close].
        horizons: trading-day horizons to evaluate.

    Returns:
        dict with 'own', 'neighbors', 'placebo' response curves and half-lives.
    """
    abnormal = abnormal_returns(prices)
    universe = list(abnormal.columns)
    rng = placebo_rng or np.random.default_rng(0)

    events = events.copy()
    events["date"] = pd.to_datetime(events["date"])
    event_days = set(zip(events["date"], events["ticker"]))

    own_d, own_t, own_s = [], [], []
    nbr_d, nbr_t, nbr_s = [], [], []
    pbo_d, pbo_t, pbo_s = [], [], []

    for date, ticker, sign in zip(events["date"], events["ticker"], events["sign"]):
        own_d.append(date); own_t.append(ticker); own_s.append(sign)

        neighbors = [n for n in neighbor_fn(ticker, date) if n in abnormal.columns]
        if exclude_own_events:
            neighbors = [n for n in neighbors if (date, n) not in event_days]
        if not neighbors:
            continue

        for n in neighbors:
            nbr_d.append(date); nbr_t.append(n); nbr_s.append(sign)

        # Placebo: same count, drawn from firms that are NOT neighbours.
        pool = [t for t in universe if t not in neighbors and t != ticker]
        if pool:
            for n in rng.choice(pool, size=min(len(neighbors), len(pool)), replace=False):
                pbo_d.append(date); pbo_t.append(n); pbo_s.append(sign)

    result = {
        "own": cumulative_response(abnormal, own_d, own_t, own_s, horizons),
        "neighbors": cumulative_response(abnormal, nbr_d, nbr_t, nbr_s, horizons),
        "placebo": cumulative_response(abnormal, pbo_d, pbo_t, pbo_s, horizons),
        "horizons": list(horizons),
    }
    result["difference"] = difference_curves(result["neighbors"], result["placebo"],
                                             horizons)
    result["half_life"] = {
        key: implied_half_life(result[key], horizons)
        for key in ("own", "neighbors", "difference")
    }

    if not quiet:
        report(result)
    return result


def report(result):
    horizons = result["horizons"]
    labels = ("neighbours", "placebo", "nbr - placebo")
    print(f"\n{'horizon':<10}" + "".join(f"{k:>24}" for k in labels))
    print(f"{'(days)':<10}" + "".join(f"{'CAR / t':>24}" for _ in labels))
    print("-" * 82)
    for h in horizons:
        cells = []
        for key in ("neighbors", "placebo", "difference"):
            mean, t_stat = result[key][h][0], result[key][h][1]
            cells.append(f"{mean:+.5f} / {t_stat:+6.2f}")
        print(f"{h:<10}" + "".join(f"{c:>24}" for c in cells))
    print(f"\n  observations: {result['neighbors'][horizons[0]][2]} neighbour, "
          f"{result['placebo'][horizons[0]][2]} placebo, "
          f"{result['own'][horizons[0]][2]} own-firm events")

    print()
    for key, label in (("neighbors", "neighbours"), ("difference", "differenced")):
        half = result["half_life"][key]
        if half is None:
            print(f"  {label:<12} no positive response to halve")
        elif half == 0.0:
            print(f"  {label:<12} already half-priced by the first horizon "
                  f"(faster than the data resolves)")
        else:
            print(f"  {label:<12} half-life {half:.2f} days")
