"""Does temporal earnings history add information beyond option prices?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from models.earnings_distribution import (build_option_event_panel,
                                          build_pre_event_features,
                                          build_standardized_event_variance,
                                          probability_metrics)


def fit_classifier(train, validation, test, features, target):
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=25, learning_rate=.05,
        l2_regularization=3, random_state=0)
    model.fit(train[features], train[target])
    # Preserve ranking for the paired out-of-sample comparison. Calibration is
    # reported separately by probability_metrics in the distribution baseline.
    return model.predict_proba(test[features])[:, 1]


def date_bootstrap(frame, left, right, target, repetitions=1000, seed=0):
    rng = np.random.default_rng(seed)
    dates = frame["anndats_act"].dt.normalize().unique()
    auc, ap = [], []
    for _ in range(repetitions):
        sampled = rng.choice(dates, len(dates), replace=True)
        indices = np.concatenate([
            frame.index[frame["anndats_act"].dt.normalize().eq(date)].to_numpy()
            for date in sampled
        ])
        truth = frame.loc[indices, target].to_numpy()
        if truth.min() == truth.max():
            continue
        auc.append(roc_auc_score(truth, left[indices])
                   - roc_auc_score(truth, right[indices]))
        ap.append(average_precision_score(truth, left[indices])
                  - average_precision_score(truth, right[indices]))
    return (np.quantile(auc, [.025, .5, .975]),
            np.quantile(ap, [.025, .5, .975]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--options", default=os.path.join(
        ROOT, "data/earnings/option_entry_straddles.parquet"))
    parser.add_argument("--standardized", default=os.path.join(
        ROOT, "data/earnings/option_standardized_terms.parquet"))
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--validation-end", default="2023-12-31")
    parser.add_argument("--tail", type=float, default=.15)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", default=os.path.join(
        ROOT, "data/earnings/option_market_predictions.parquet"))
    args = parser.parse_args()

    raw = pd.read_parquet(args.events)
    frame, temporal_features = build_pre_event_features(raw)
    frame = build_option_event_panel(frame, pd.read_parquet(args.options))
    if os.path.exists(args.standardized):
        frame = frame.merge(build_standardized_event_variance(
            pd.read_parquet(args.standardized)), on="event_id", how="left",
            validate="one_to_one")
    frame = frame[(frame["implied_move"].between(.005, .80))
                  & (frame["option_spread_pct"].between(0, 1.5))].copy()
    frame["tail"] = frame["abs_reaction_1d"] >= args.tail
    frame["exceeded_implied"] = frame["abs_reaction_1d"] > frame["implied_move"]

    train = frame[frame["anndats_act"] <= args.train_end].copy()
    validation = frame[(frame["anndats_act"] > args.train_end)
                       & (frame["anndats_act"] <= args.validation_end)].copy()
    test = frame[frame["anndats_act"] > args.validation_end].copy().reset_index(drop=True)
    print(f"option-covered events: train {len(train):,} | validation {len(validation):,} "
          f"| test {len(test):,}; test tail {test['tail'].mean():.1%}")

    option_features = [
        "implied_move", "implied_move_ask", "option_mean_iv", "option_put_skew",
        "option_delta_balance", "option_spread_pct", "option_dte",
        "option_days_after_event", "option_event_sigma",
        "option_diffusive_variance", "option_term_points",
        "std_event_expected_abs", "std_event_sigma", "std_diffusive_variance",
        "std_10d_straddle_pct", "std_10d_iv",
        "log_call_oi", "log_put_oi",
        "log_call_volume", "log_put_volume",
    ]
    option_features = [column for column in option_features if column in frame]
    temporal_features = [column for column in temporal_features if column in frame]
    predictions = {}
    for name, features in (
        ("options only", option_features),
        ("temporal only", temporal_features),
        ("options + temporal", option_features + temporal_features),
    ):
        probability = fit_classifier(train, validation, test, features, "tail")
        predictions[name] = probability
        result = probability_metrics(test["tail"], probability, name)
        print(f"{name:<20} AUC {result['auc']:.3f} | AP {result['ap']:.3f} | "
              f"top10 lift {result['top_decile_lift']:.2f}x")

    auc_ci, ap_ci = date_bootstrap(
        test, predictions["options + temporal"], predictions["options only"],
        "tail", args.bootstrap)
    print("options+temporal minus options-only date bootstrap")
    print(f"AUC 95% CI [{auc_ci[0]:+.4f}, {auc_ci[2]:+.4f}], median {auc_ci[1]:+.4f}")
    print(f"AP  95% CI [{ap_ci[0]:+.4f}, {ap_ci[2]:+.4f}], median {ap_ci[1]:+.4f}")

    print("\nabsolute-move regression")
    if "std_event_expected_abs" in test:
        valid = test["std_event_expected_abs"].notna()
        print(f"standardized event prior coverage {valid.mean():.1%} | "
              f"MAE {mean_absolute_error(test.loc[valid, 'abs_reaction_1d'], test.loc[valid, 'std_event_expected_abs']):.4f} | "
              f"correlation {test.loc[valid, ['abs_reaction_1d', 'std_event_expected_abs']].corr().iloc[0, 1]:+.3f}")
    for name, features in (
        ("market straddle", []),
        ("options model", option_features),
        ("options + temporal", option_features + temporal_features),
    ):
        if not features:
            prediction = test["implied_move"].to_numpy()
        else:
            model = HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=250, max_leaf_nodes=25,
                learning_rate=.05, l2_regularization=3, random_state=0)
            model.fit(train[features], train["abs_reaction_1d"])
            prediction = np.maximum(model.predict(test[features]), 0)
        mae = mean_absolute_error(test["abs_reaction_1d"], prediction)
        correlation = np.corrcoef(test["abs_reaction_1d"], prediction)[0, 1]
        print(f"{name:<20} MAE {mae:.4f} | correlation {correlation:+.3f}")
        predictions[f"move_{name}"] = prediction

    output = test[["event_id", "permno", "secid", "ticker", "anndats_act",
                   "reaction_1d", "abs_reaction_1d", "implied_move",
                   "implied_move_ask", "exceeded_implied"]].copy()
    for name, values in predictions.items():
        output[name.replace(" ", "_")] = values
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
