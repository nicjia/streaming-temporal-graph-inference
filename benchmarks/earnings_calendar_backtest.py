"""Historical same-strike earnings calendar-spread evaluation."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from models.earnings_distribution import (build_calendar_pairs,
                                          build_option_event_panel,
                                          build_pre_event_features)


def summarize(name, frame, pnl, debit):
    returns = frame[pnl] / frame[debit]
    date_return = returns.groupby(pd.to_datetime(frame["anndats_act"]).dt.normalize()).mean()
    standard_error = date_return.std(ddof=1) / np.sqrt(len(date_return))
    t_stat = date_return.mean() / standard_error if standard_error > 0 else np.nan
    print(f"{name:<25} n {len(frame):,} | mean {returns.mean():+.1%} | "
          f"median {returns.median():+.1%} | win {(returns > 0).mean():.1%} | "
          f"date-cluster t {t_stat:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--options", default=os.path.join(
        ROOT, "data/earnings/option_entry_straddles.parquet"))
    parser.add_argument("--marks", default=os.path.join(
        ROOT, "data/earnings/calendar_post_event_marks.parquet"))
    parser.add_argument("--pairs", default=os.path.join(
        ROOT, "data/earnings/calendar_pairs.parquet"))
    parser.add_argument("--out", default=os.path.join(
        ROOT, "data/earnings/calendar_backtest.parquet"))
    args = parser.parse_args()

    raw = pd.read_parquet(args.events)
    events, temporal_features = build_pre_event_features(raw)
    straddles = pd.read_parquet(args.options)
    market = build_option_event_panel(events, straddles)
    if os.path.exists(args.pairs):
        calendars = pd.read_parquet(args.pairs)
    else:
        calendars = build_calendar_pairs(straddles)
        calendars = calendars.merge(
            events[["event_id", "reaction_date"]], on="event_id", validate="one_to_one")
    frame = calendars.merge(market, on="event_id", suffixes=("", "_event"),
                            validate="one_to_one")
    frame = frame.merge(pd.read_parquet(args.marks), on="event_id",
                        validate="one_to_one")

    frame["exit_credit_mid"] = (
        frame["back_call_bid"] + frame["back_call_ask"]
        + frame["back_put_bid"] + frame["back_put_ask"]
        - frame["front_call_bid"] - frame["front_call_ask"]
        - frame["front_put_bid"] - frame["front_put_ask"]) / 2
    frame["exit_credit_conservative"] = (
        frame["back_call_bid"] + frame["back_put_bid"]
        - frame["front_call_ask"] - frame["front_put_ask"])
    frame["pnl_mid"] = frame["exit_credit_mid"] - frame["entry_debit_mid"]
    frame["pnl_conservative"] = (
        frame["exit_credit_conservative"] - frame["entry_debit_conservative"])
    frame["entry_cost_width"] = (
        frame["entry_debit_conservative"] - frame["entry_debit_mid"])
    frame["front_straddle_mid"] = (
        frame["call_bid_front"] + frame["call_ask_front"]
        + frame["put_bid_front"] + frame["put_ask_front"]) / 2
    frame["back_straddle_mid"] = (
        frame["call_bid_back"] + frame["call_ask_back"]
        + frame["put_bid_back"] + frame["put_ask_back"]) / 2
    frame["calendar_debit_pct"] = frame["entry_debit_mid"] / frame["underlying_close_front"]
    frame["front_implied_move"] = frame["front_straddle_mid"] / frame["underlying_close_front"]
    frame["calendar_term_ratio"] = frame["back_straddle_mid"] / frame["front_straddle_mid"]
    frame["calendar_cost_ratio"] = frame["entry_cost_width"] / frame["entry_debit_mid"]
    frame["calendar_iv_slope"] = (
        (frame["call_iv_back"] + frame["put_iv_back"])
        - (frame["call_iv_front"] + frame["put_iv_front"])) / 2
    frame["calendar_days_between"] = (
        pd.to_datetime(frame["exdate_back"]) - pd.to_datetime(frame["exdate_front"])).dt.days
    frame["positive_conservative"] = frame["pnl_conservative"] > 0
    frame["positive_mid"] = frame["pnl_mid"] > 0

    # Every filter is observable at entry. Avoid tiny debits whose percentage
    # returns are dominated by one quote tick.
    frame = frame[(frame["entry_debit_mid"] >= .10)
                  & frame["calendar_cost_ratio"].between(0, 5)
                  & frame["front_implied_move"].between(.005, .80)].copy()
    train = frame[frame["anndats_act"] <= "2022-12-31"].copy()
    validation = frame[(frame["anndats_act"] > "2022-12-31")
                       & (frame["anndats_act"] <= "2023-12-31")].copy()
    test = frame[frame["anndats_act"] > "2023-12-31"].copy()
    print(f"calendar events: train {len(train):,} | validation {len(validation):,} | "
          f"test {len(test):,}")
    summarize("all trades, midpoint", test, "pnl_mid", "entry_debit_mid")
    summarize("all trades, executable", test, "pnl_conservative",
              "entry_debit_conservative")

    option_features = [
        "implied_move", "option_event_sigma", "option_diffusive_variance",
        "option_put_skew", "option_spread_pct", "calendar_debit_pct",
        "front_implied_move", "calendar_term_ratio", "calendar_cost_ratio",
        "calendar_iv_slope", "calendar_days_between", "entry_debit_mid",
        "log_call_oi", "log_put_oi", "log_call_volume", "log_put_volume",
    ]
    option_features = [column for column in option_features if column in frame]
    temporal_features = [column for column in temporal_features if column in frame]
    for target, label in (("positive_mid", "midpoint"),
                          ("positive_conservative", "executable")):
        print(f"\nselection target: {label}")
        for name, features in (("options only", option_features),
                               ("options + temporal", option_features + temporal_features)):
            model = HistGradientBoostingClassifier(
                max_iter=250, max_leaf_nodes=25, learning_rate=.05,
                l2_regularization=3, random_state=0)
            model.fit(train[features], train[target])
            score = model.predict_proba(test[features])[:, 1]
            auc = roc_auc_score(test[target], score)
            ap = average_precision_score(test[target], score)
            selected = test.loc[score >= np.quantile(score, .80)].copy()
            print(f"{name:<25} positive-PnL AUC {auc:.3f} | AP {ap:.3f}")
            summarize(name + " top20 mid", selected, "pnl_mid", "entry_debit_mid")
            summarize(name + " top20 exec", selected, "pnl_conservative",
                      "entry_debit_conservative")
            if target == "positive_mid":
                gross = (selected["pnl_mid"] / selected["entry_debit_mid"]).mean()
                quoted_shortfall = ((selected["pnl_mid"]
                                     - selected["pnl_conservative"])
                                    / selected["entry_debit_mid"]).mean()
                threshold = gross / quoted_shortfall if quoted_shortfall > 0 else np.nan
                print(f"{'execution threshold':<25} {threshold:.1%} of quoted "
                      "four-leg NBBO shortfall")
            column = f"{name.replace(' ', '_')}_{label}_score"
            frame.loc[test.index, column] = score

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
