"""Can own history and the recent earnings regime predict move tails?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_pinball_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from models.earnings_distribution import build_pre_event_features, probability_metrics


def calibrate(validation_probability, validation_target, test_probability):
    """Platt calibration fit only on the middle chronological period."""
    transform = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6)
                                 / np.clip(1 - x, 1e-6, 1 - 1e-6))
    model = LogisticRegression(C=1).fit(
        transform(validation_probability).reshape(-1, 1), validation_target)
    return model.predict_proba(transform(test_probability).reshape(-1, 1))[:, 1]


def print_metrics(result):
    print(f"{result['name']:<25} AUC {result['auc']:.3f} | AP {result['ap']:.3f} | "
          f"Brier {result['brier']:.4f} | log {result['log_loss']:.3f} | "
          f"top10 recall {result['top_decile_recall']:.1%} / "
          f"lift {result['top_decile_lift']:.2f}x | ECE {result['ece']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--train-end", default="2021-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument("--tail", type=float, default=.15)
    parser.add_argument("--out", default=os.path.join(
        ROOT, "data/earnings/baseline_predictions.parquet"))
    parser.add_argument("--skip-quantiles", action="store_true")
    args = parser.parse_args()

    raw = pd.read_parquet(args.data)
    frame, features = build_pre_event_features(raw)
    train = frame[frame["anndats_act"] <= args.train_end].copy()
    validation = frame[(frame["anndats_act"] > args.train_end)
                       & (frame["anndats_act"] <= args.validation_end)].copy()
    test = frame[frame["anndats_act"] > args.validation_end].copy()
    target_name = "target"
    for split in (train, validation, test):
        split[target_name] = split["abs_reaction_1d"] >= args.tail
    print(f"tail >= {args.tail:.0%}: train {len(train):,} | validation {len(validation):,} | "
          f"test {len(test):,} ({test[target_name].mean():.1%} positive)")

    constant = np.full(len(test), train[target_name].mean())
    print_metrics(probability_metrics(test[target_name], constant, "unconditional"))

    own = test["own_tail10_rate_8"].fillna(train[target_name].mean()).to_numpy()
    # The rolling rate is for 10% moves; calibrating it on validation makes it
    # a valid score for whichever tail threshold this run requests.
    own_validation = validation["own_tail10_rate_8"].fillna(
        train[target_name].mean()).to_numpy()
    own = calibrate(own_validation, validation[target_name], own)
    print_metrics(probability_metrics(test[target_name], own, "own recent earnings"))

    regime_column = "regime_tail15_30d" if args.tail >= .15 else "regime_tail10_30d"
    regime_validation = validation[regime_column].fillna(
        train[target_name].mean()).to_numpy()
    regime = test[regime_column].fillna(train[target_name].mean()).to_numpy()
    regime = calibrate(regime_validation, validation[target_name], regime)
    print_metrics(probability_metrics(test[target_name], regime, "recent market regime"))

    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=31, learning_rate=.05,
        l2_regularization=2, random_state=0)
    model.fit(train[features], train[target_name])
    validation_probability = model.predict_proba(validation[features])[:, 1]
    test_probability = model.predict_proba(test[features])[:, 1]
    calibrated = calibrate(validation_probability, validation[target_name], test_probability)
    print_metrics(probability_metrics(test[target_name], calibrated,
                                      "non-graph nonlinear"))

    quantile_predictions = {}
    if not args.skip_quantiles:
        print("\nabsolute-move quantiles")
    for quantile in (() if args.skip_quantiles else (.50, .75, .90, .95)):
        regressor = HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile, max_iter=200,
            max_leaf_nodes=31, learning_rate=.05, l2_regularization=2,
            random_state=0)
        regressor.fit(train[features], train["abs_reaction_1d"])
        prediction = np.maximum(regressor.predict(test[features]), 0)
        quantile_predictions[f"q{int(quantile * 100)}"] = prediction
        coverage = float((test["abs_reaction_1d"].to_numpy() <= prediction).mean())
        loss = mean_pinball_loss(test["abs_reaction_1d"], prediction, alpha=quantile)
        unconditional_q = float(train["abs_reaction_1d"].quantile(quantile))
        baseline_loss = mean_pinball_loss(
            test["abs_reaction_1d"], np.full(len(test), unconditional_q), alpha=quantile)
        print(f"q{quantile:.2f}: coverage {coverage:.1%} | pinball {loss:.5f} "
              f"vs unconditional {baseline_loss:.5f} | "
              f"improvement {(baseline_loss - loss) / baseline_loss:+.1%}")

    output = test[["event_id", "permno", "secid", "ticker", "oftic",
                   "anndats_act", "reaction_1d", "abs_reaction_1d"]].copy()
    output["target"] = test[target_name].to_numpy()
    output["unconditional"] = constant
    output["own_history"] = own
    output["market_regime"] = regime
    output["non_graph"] = calibrated
    for name, values in quantile_predictions.items():
        output[name] = values
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
