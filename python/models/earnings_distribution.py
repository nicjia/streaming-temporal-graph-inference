"""Leakage-safe features and metrics for earnings-move distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)


def build_option_event_panel(events: pd.DataFrame,
                             straddles: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest-expiry, nearest-ATM pre-event straddle to each event."""
    options = straddles.copy()
    for column in ("fpedats", "anndats_act", "entry_date", "exdate"):
        if column in options:
            options[column] = pd.to_datetime(options[column])
    numeric = [
        "underlying_close", "strike", "call_bid", "call_ask", "put_bid",
        "put_ask", "call_iv", "put_iv", "call_delta", "put_delta",
        "call_oi", "put_oi", "call_volume", "put_volume",
    ]
    for column in numeric:
        if column in options:
            options[column] = pd.to_numeric(options[column], errors="coerce")
    event_keys = (["event_id"] if "event_id" in options
                  else ["secid", "fpedats", "anndats_act"])
    options = options.sort_values(
        event_keys + ["exdate", "distance_rank"], kind="stable")
    options = options.drop_duplicates(event_keys + ["exdate"], keep="first")
    options["option_mean_iv"] = (options["call_iv"] + options["put_iv"]) / 2
    options["option_dte"] = (options["exdate"] - options["entry_date"]).dt.days

    # All expiries after the announcement contain the same scheduled jump.
    # Regressing total variance IV^2*T on maturity separates the common
    # intercept (event variance) from the post-event diffusive slope.
    term_source = options[event_keys + ["option_dte", "option_mean_iv"]].copy()
    term_source["x"] = term_source["option_dte"] / 365.0
    term_source["y"] = term_source["option_mean_iv"] ** 2 * term_source["x"]
    term_source = term_source.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    term_source = term_source[term_source["x"] > 0]
    term_source["xx"] = term_source["x"] ** 2
    term_source["xy"] = term_source["x"] * term_source["y"]
    term = term_source.groupby(event_keys, as_index=False).agg(
        option_term_points=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
        sxx=("xx", "sum"), sxy=("xy", "sum"))
    denominator = term["option_term_points"] * term["sxx"] - term["sx"] ** 2
    slope = ((term["option_term_points"] * term["sxy"] - term["sx"] * term["sy"])
             / denominator.replace(0, np.nan))
    intercept = (term["sy"] - slope * term["sx"]) / term["option_term_points"]
    term["option_event_variance"] = intercept.clip(lower=0)
    term["option_event_sigma"] = np.sqrt(term["option_event_variance"])
    term["option_diffusive_variance"] = slope.clip(lower=0)
    term = term.drop(columns=["sx", "sy", "sxx", "sxy"])
    options = options.sort_values(
        event_keys + ["exdate"], kind="stable").drop_duplicates(
            event_keys, keep="first")
    options["straddle_mid"] = (
        options["call_bid"] + options["call_ask"]
        + options["put_bid"] + options["put_ask"]) / 2
    options["straddle_bid"] = options["call_bid"] + options["put_bid"]
    options["straddle_ask"] = options["call_ask"] + options["put_ask"]
    options["implied_move"] = options["straddle_mid"] / options["underlying_close"]
    options["implied_move_ask"] = options["straddle_ask"] / options["underlying_close"]
    options["option_spread_pct"] = (
        options["straddle_ask"] - options["straddle_bid"]
    ) / options["straddle_mid"].replace(0, np.nan)
    options["option_put_skew"] = options["put_iv"] - options["call_iv"]
    options["option_delta_balance"] = options["call_delta"] + options["put_delta"]
    options["option_days_after_event"] = (
        options["exdate"] - options["anndats_act"]).dt.days
    options = options.merge(term, on=event_keys, how="left", validate="one_to_one")
    for side in ("call", "put"):
        if f"{side}_oi" in options:
            options[f"log_{side}_oi"] = np.log1p(options[f"{side}_oi"].clip(lower=0))
        if f"{side}_volume" in options:
            options[f"log_{side}_volume"] = np.log1p(
                options[f"{side}_volume"].clip(lower=0))
    keys = event_keys
    keep = keys + [column for column in options.columns if column.startswith("option_")]
    keep += ["entry_date", "exdate", "underlying_close", "strike",
             "straddle_mid", "straddle_bid", "straddle_ask", "implied_move",
             "implied_move_ask", "call_iv", "put_iv", "log_call_oi",
             "log_put_oi", "log_call_volume", "log_put_volume"]
    keep = list(dict.fromkeys(column for column in keep if column in options))
    return events.merge(options[keep], on=keys, how="inner", validate="one_to_one")


def build_calendar_pairs(straddles: pd.DataFrame,
                         max_front_days: int = 5) -> pd.DataFrame:
    """Select same-strike front/back ATM double-calendar straddles."""
    source = straddles.copy()
    for column in ("anndats_act", "entry_date", "exdate"):
        source[column] = pd.to_datetime(source[column])
    numeric = [column for column in source if column.endswith(
        ("_bid", "_ask", "_iv", "_delta", "_oi", "_volume", "_optionid"))]
    numeric += ["strike", "underlying_close", "distance_rank"]
    for column in set(numeric):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["days_after_event"] = (source["exdate"] - source["anndats_act"]).dt.days

    rows = []
    for event_id, group in source.groupby("event_id", sort=False):
        expiries = np.sort(group.loc[
            group["days_after_event"].between(0, max_front_days), "exdate"].unique())
        if not len(expiries):
            continue
        front_expiry = pd.Timestamp(expiries[0])
        back_expiries = np.sort(group.loc[
            (group["exdate"] >= front_expiry + pd.Timedelta(days=5))
            & (group["exdate"] <= front_expiry + pd.Timedelta(days=10)), "exdate"].unique())
        if not len(back_expiries):
            continue
        front = group[group["exdate"].eq(front_expiry)]
        candidates = []
        for back_expiry in back_expiries:
            back = group[group["exdate"].eq(pd.Timestamp(back_expiry))]
            common = front.merge(back, on="strike", suffixes=("_front", "_back"))
            if len(common):
                common["pair_rank"] = (common["distance_rank_front"]
                                       + common["distance_rank_back"])
                candidates.append(common)
        if not candidates:
            continue
        pair = pd.concat(candidates, ignore_index=True).sort_values(
            ["pair_rank", "exdate_back", "strike"], kind="stable").iloc[0]
        item = {"event_id": int(event_id), "strike": float(pair["strike"])}
        for column, value in pair.items():
            if column != "strike":
                item[column] = value
        front_mid = ((item["call_bid_front"] + item["call_ask_front"]
                      + item["put_bid_front"] + item["put_ask_front"]) / 2)
        back_mid = ((item["call_bid_back"] + item["call_ask_back"]
                     + item["put_bid_back"] + item["put_ask_back"]) / 2)
        item["entry_debit_mid"] = back_mid - front_mid
        item["entry_debit_conservative"] = (
            item["call_ask_back"] + item["put_ask_back"]
            - item["call_bid_front"] - item["put_bid_front"])
        rows.append(item)
    result = pd.DataFrame(rows)
    if len(result):
        result = result[(result["entry_debit_mid"] > 0)
                        & (result["entry_debit_conservative"] > 0)].reset_index(drop=True)
    return result


def build_standardized_event_variance(terms: pd.DataFrame) -> pd.DataFrame:
    """Separate scheduled jump variance from standardized ATM term structure."""
    source = terms.copy()
    numeric = ["days", "forward_price", "call_premium", "put_premium",
               "call_iv", "put_iv", "call_delta", "put_delta"]
    for column in numeric:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["mean_iv"] = (source["call_iv"] + source["put_iv"]) / 2
    source["maturity"] = source["days"] / 365.0
    source["total_variance"] = source["mean_iv"] ** 2 * source["maturity"]
    short = source[source["days"] <= 60].dropna(
        subset=["maturity", "total_variance"]).copy()
    short["xx"] = short["maturity"] ** 2
    short["xy"] = short["maturity"] * short["total_variance"]
    result = short.groupby("event_id", as_index=False).agg(
        std_term_points=("maturity", "size"), sx=("maturity", "sum"),
        sy=("total_variance", "sum"), sxx=("xx", "sum"), sxy=("xy", "sum"))
    denominator = result["std_term_points"] * result["sxx"] - result["sx"] ** 2
    slope = ((result["std_term_points"] * result["sxy"]
              - result["sx"] * result["sy"]) / denominator.replace(0, np.nan))
    intercept = (result["sy"] - slope * result["sx"]) / result["std_term_points"]
    result["std_event_variance_raw"] = intercept
    result["std_event_variance"] = intercept.clip(lower=0)
    result["std_event_sigma"] = np.sqrt(result["std_event_variance"])
    result["std_event_expected_abs"] = result["std_event_sigma"] * np.sqrt(2 / np.pi)
    result["std_diffusive_variance"] = slope.clip(lower=0)
    result = result.drop(columns=["sx", "sy", "sxx", "sxy"])
    ten = source[source["days"].eq(10)].copy()
    ten["std_10d_straddle_pct"] = (
        ten["call_premium"] + ten["put_premium"]) / ten["forward_price"]
    ten["std_10d_iv"] = ten["mean_iv"]
    ten = ten[["event_id", "std_10d_straddle_pct", "std_10d_iv"]].drop_duplicates(
        "event_id")
    return result.merge(ten, on="event_id", how="outer", validate="one_to_one")


def add_event_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine I/B/E/S Eastern date/time into a sortable event timestamp."""
    out = frame.copy()
    clock = out["anntims_act"].astype("string").fillna("12:00:00")
    out["event_timestamp"] = (
        pd.to_datetime(out["anndats_act"].dt.strftime("%Y-%m-%d") + " " + clock)
        .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    return out


def _prior_rolling(group: pd.Series, window: int, statistic: str,
                   minimum: int = 2) -> pd.Series:
    prior = group.shift(1).rolling(window, min_periods=minimum)
    if statistic == "mean":
        return prior.mean()
    if statistic == "std":
        return prior.std()
    if statistic == "max":
        return prior.max()
    if statistic == "median":
        return prior.median()
    raise ValueError(statistic)


def _causal_time_ewm(values, times,
                     half_life_days: float) -> np.ndarray:
    """Irregular-time EWM emitted before admitting each current observation."""
    data = np.asarray(values, dtype=float)
    clock = np.asarray(times, dtype=np.int64)
    output = np.full(len(data), np.nan)
    weighted_sum = 0.0
    weight = 0.0
    previous = None
    decay_scale = np.log(2.0) / half_life_days
    for index, (value, timestamp) in enumerate(zip(data, clock)):
        if previous is not None:
            elapsed = max(float(timestamp - previous), 0.0)
            decay = np.exp(-decay_scale * elapsed)
            weighted_sum *= decay
            weight *= decay
        if weight > 0:
            output[index] = weighted_sum / weight
        if np.isfinite(value):
            weighted_sum += value
            weight += 1.0
        previous = timestamp
    return output


def build_pre_event_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create own-history and market-regime features using prior events only."""
    out = add_event_timestamp(frame).sort_values(
        ["event_timestamp", "event_id"], kind="stable").reset_index(drop=True)
    out["abs_reaction_1d"] = out["reaction_1d"].abs()
    out["tail_10"] = out["abs_reaction_1d"] >= .10
    out["tail_15"] = out["abs_reaction_1d"] >= .15
    out["tail_20"] = out["abs_reaction_1d"] >= .20

    by_company = out.groupby("permno", sort=False, group_keys=False)
    out["own_events_seen"] = by_company.cumcount()
    for window in (4, 8, 12):
        out[f"own_abs_mean_{window}"] = by_company["abs_reaction_1d"].transform(
            lambda x, w=window: _prior_rolling(x, w, "mean"))
        out[f"own_abs_max_{window}"] = by_company["abs_reaction_1d"].transform(
            lambda x, w=window: _prior_rolling(x, w, "max"))
        out[f"own_tail10_rate_{window}"] = by_company["tail_10"].transform(
            lambda x, w=window: _prior_rolling(x.astype(float), w, "mean"))
    out["own_last_abs"] = by_company["abs_reaction_1d"].shift(1)
    out["own_last_signed"] = by_company["reaction_1d"].shift(1)
    out["own_last_surprise_z"] = by_company["eps_surprise_z"].shift(1)
    out["own_last_post_vol"] = by_company["post_5d_vol"].shift(1)
    # Earnings arrive irregularly and regimes do not respect a fixed count of
    # quarters. These causal half-life features implement the user's intended
    # recency weighting directly in calendar time.
    company_positions = list(out.groupby("permno", sort=False).indices.values())
    event_clock = out["event_timestamp"].to_numpy(
        dtype="datetime64[D]").astype(np.int64)
    for half_life in (90, 180, 365):
        for source, stem in (
            ("abs_reaction_1d", "abs"), ("tail_10", "tail10"),
            ("reaction_1d", "signed"), ("eps_surprise_z", "surprise"),
        ):
            feature = np.full(len(out), np.nan)
            source_values = out[source].to_numpy(dtype=float)
            for positions in company_positions:
                feature[positions] = _causal_time_ewm(
                    source_values[positions], event_clock[positions], half_life)
            out[f"own_ewm_{stem}_{half_life}d"] = feature

    # A conservative regime signal: every same-day event is excluded, even if
    # its announcement time preceded the focal release. This makes the daily
    # lag auditable and cannot leak one company's reaction into another.
    out["event_date"] = out["anndats_act"].dt.normalize()
    daily = out.groupby("event_date").agg(
        regime_abs=("abs_reaction_1d", "mean"),
        regime_tail10=("tail_10", "mean"),
        regime_tail15=("tail_15", "mean"),
        regime_events=("event_id", "size"),
    ).sort_index()
    complete = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    complete["regime_events"] = complete["regime_events"].fillna(0)
    for column in ("regime_abs", "regime_tail10", "regime_tail15"):
        weighted = (complete[column].fillna(0)
                    * complete["regime_events"])
        for days in (30, 90, 180):
            numerator = weighted.shift(1).rolling(days, min_periods=5).sum()
            denominator = complete["regime_events"].shift(1).rolling(
                days, min_periods=5).sum()
            complete[f"{column}_{days}d"] = numerator / denominator.replace(0, np.nan)
    regime_columns = [column for column in complete if column.endswith("d")]
    complete = complete[regime_columns]
    out = out.merge(complete, left_on="event_date", right_index=True, how="left")

    out["consensus_dispersion"] = (
        out["stdev"].abs() / np.maximum(out["meanest"].abs(), .05))
    out["log_numest"] = np.log1p(out["numest"].clip(lower=0))
    out["log_prior_events"] = np.log1p(out["own_events_seen"])
    out["month_sin"] = np.sin(2 * np.pi * out["anndats_act"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["anndats_act"].dt.month / 12)
    out["after_close"] = out["announcement_session"].eq("after_close").astype(float)

    base = [
        "pre_20d_vol", "pre_5d_return", "consensus_dispersion", "log_numest",
        "forecast_age_days", "log_prior_events", "month_sin", "month_cos",
        "after_close",
    ]
    own = [column for column in out if column.startswith("own_")
           and column not in ("own_events_seen",)]
    return out, base + regime_columns + own


def build_supply_event_features(frame: pd.DataFrame, supply: pd.DataFrame,
                                company_map: pd.DataFrame,
                                shuffle_neighbors: bool = False,
                                seed: int = 0) -> pd.DataFrame:
    """Aggregate only already-observed earnings from active supply neighbors.

    The focal event's relationship set is resolved at its announcement date.
    Neighbor reactions are admitted only when their reaction trading date is
    strictly earlier than the focal announcement date, so an after-close
    release cannot borrow the next day's still-unobserved return.
    """
    mapping = dict(zip(company_map["company_id"].astype(str),
                       company_map["permno"].astype(int)))
    links = supply.copy()
    links["supplier"] = links["supplier_id"].astype(str).map(mapping)
    links["customer"] = links["customer_id"].astype(str).map(mapping)
    links = links.dropna(subset=["supplier", "customer", "start_"]).copy()
    links["supplier"] = links["supplier"].astype(int)
    links["customer"] = links["customer"].astype(int)
    links["start_"] = pd.to_datetime(links["start_"]).to_numpy(dtype="datetime64[D]")
    links["end_"] = pd.to_datetime(links["end_"], errors="coerce").fillna(
        pd.Timestamp("2100-01-01")).to_numpy(dtype="datetime64[D]")
    links = links.drop_duplicates(["supplier", "customer", "start_", "end_"])

    adjacency: dict[int, list[tuple[int, np.datetime64, np.datetime64, int]]] = {}
    for row in links.itertuples(index=False):
        adjacency.setdefault(int(row.supplier), []).append(
            (int(row.customer), row.start_, row.end_, 1))
        adjacency.setdefault(int(row.customer), []).append(
            (int(row.supplier), row.start_, row.end_, -1))
    if shuffle_neighbors:
        rng = np.random.default_rng(seed)
        universe = np.asarray(sorted(adjacency), dtype=int)
        # Preserve each focal node's dated degree and direction sequence while
        # replacing only counterpart identity.
        for focal, entries in adjacency.items():
            replacements = rng.choice(universe[universe != focal], len(entries),
                                      replace=True)
            adjacency[focal] = [(int(replacement), start, end, direction)
                                for replacement, (_, start, end, direction)
                                in zip(replacements, entries)]

    ordered = frame.sort_values(["reaction_date", "event_id"], kind="stable")
    histories = {}
    for permno, group in ordered.groupby("permno", sort=False):
        histories[int(permno)] = {
            "date": group["reaction_date"].to_numpy(dtype="datetime64[D]"),
            "abs": group["abs_reaction_1d"].to_numpy(dtype=float),
            "signed": group["reaction_1d"].to_numpy(dtype=float),
            "tail": group["tail_15"].to_numpy(dtype=float),
            "surprise": group["eps_surprise_z"].to_numpy(dtype=float),
        }

    rows = []
    for event in frame.itertuples(index=False):
        date = np.datetime64(event.anndats_act, "D")
        aggregate = {30: {"abs": [], "signed": [], "tail": [], "surprise": []},
                     90: {"abs": [], "signed": [], "tail": [], "surprise": []}}
        active = set()
        directions = []
        latest_date = None
        for neighbor, start, end, direction in adjacency.get(int(event.permno), []):
            if not (start <= date <= end) or neighbor in active:
                continue
            active.add(neighbor); directions.append(direction)
            history = histories.get(neighbor)
            if history is None:
                continue
            right = int(np.searchsorted(history["date"], date, side="left"))
            if right == 0:
                continue
            observed_date = history["date"][right - 1]
            if latest_date is None or observed_date > latest_date:
                latest_date = observed_date
            for days in (30, 90):
                left = int(np.searchsorted(history["date"], date - np.timedelta64(days, "D"),
                                           side="left"))
                for key in aggregate[days]:
                    aggregate[days][key].extend(history[key][left:right])
        item = {
            "event_id": int(event.event_id),
            "graph_active_neighbors": len(active),
            "graph_supplier_share": (float(np.mean(np.asarray(directions) < 0))
                                     if directions else np.nan),
            "graph_days_since_event": (float((date - latest_date).astype(int))
                                       if latest_date is not None else np.nan),
        }
        for days in (30, 90):
            values = aggregate[days]
            item[f"graph_events_{days}d"] = len(values["abs"])
            for key in ("abs", "signed", "tail", "surprise"):
                item[f"graph_{key}_{days}d"] = (
                    float(np.mean(values[key])) if values[key] else np.nan)
        rows.append(item)
    return pd.DataFrame(rows)


def build_industry_event_features(frame: pd.DataFrame,
                                  classifications=("sic2", "sic3", "naics2", "naics3"),
                                  prefix="industry") -> pd.DataFrame:
    """Aggregate prior earnings through dated industry/theme membership."""
    source = frame.copy()
    source["event_date"] = pd.to_datetime(source["anndats_act"]).dt.normalize()
    outputs = source[["event_id"]].copy()
    for classification in classifications:
        valid = source.dropna(subset=[classification]).copy()
        daily = valid.groupby([classification, "event_date"]).agg(
            count=("event_id", "size"),
            abs_sum=("abs_reaction_1d", "sum"),
            signed_sum=("reaction_1d", "sum"),
            tail_sum=("tail_15", "sum"),
            surprise_sum=("eps_surprise_z", "sum"),
        ).reset_index()
        feature_parts = []
        for code, group in daily.groupby(classification, sort=False):
            group = group.sort_values("event_date").set_index("event_date")
            block = pd.DataFrame(index=group.index)
            for days in (30, 90):
                rolling = group[["count", "abs_sum", "signed_sum", "tail_sum",
                                 "surprise_sum"]].rolling(
                                     f"{days}D", closed="left", min_periods=1).sum()
                denominator = rolling["count"].replace(0, np.nan)
                stem = f"{prefix}_{classification}_{days}d"
                block[f"{stem}_events"] = rolling["count"]
                block[f"{stem}_abs"] = rolling["abs_sum"] / denominator
                block[f"{stem}_signed"] = rolling["signed_sum"] / denominator
                block[f"{stem}_tail"] = rolling["tail_sum"] / denominator
                block[f"{stem}_surprise"] = rolling["surprise_sum"] / denominator
            block[classification] = code
            feature_parts.append(block.reset_index())
        features = pd.concat(feature_parts, ignore_index=True)
        event_keys = source[["event_id", "event_date", classification]]
        event_features = event_keys.merge(
            features, on=["event_date", classification], how="left").drop(
                columns=["event_date", classification])
        outputs = outputs.merge(event_features, on="event_id", how="left")
    return outputs


def probability_metrics(target, probability, name: str) -> dict[str, float]:
    target = np.asarray(target, dtype=bool)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    cutoff = np.quantile(probability, .90)
    selected = probability >= cutoff
    prevalence = float(target.mean())
    bins = pd.qcut(probability, 10, duplicates="drop")
    calibration = pd.DataFrame({"target": target, "probability": probability,
                                "bin": bins}).dropna(subset=["bin"]).groupby(
                                    "bin", observed=True).agg(
        observed=("target", "mean"), predicted=("probability", "mean"),
        count=("target", "size"))
    ece = (float((np.abs(calibration["observed"] - calibration["predicted"])
                  * calibration["count"]).sum() / calibration["count"].sum())
           if len(calibration) else float(np.abs(prevalence - probability.mean())))
    return {
        "name": name,
        "prevalence": prevalence,
        "auc": float(roc_auc_score(target, probability)),
        "ap": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "top_decile_recall": float(target[selected].sum() / max(target.sum(), 1)),
        "top_decile_lift": float(target[selected].mean() / max(prevalence, 1e-12)),
        "ece": ece,
    }
