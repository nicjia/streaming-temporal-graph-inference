"""Does typed temporal attention add tail information beyond option prices?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from benchmarks.earnings_tgat_tail import (build_self_event_graph, predict,
                                           timestamp_seconds, train_model)
from models.earnings_distribution import (build_option_event_panel,
                                          build_pre_event_features,
                                          build_standardized_event_variance)


def report(name, target, score):
    print(f"{name:<25} AUC {roc_auc_score(target, score):.3f} | "
          f"AP {average_precision_score(target, score):.3f}")


def date_bootstrap(frame, baseline, challenger, repetitions=500, seed=0):
    dates = frame["anndats_act"].dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(dates)
    rows = {date: np.flatnonzero(dates == date) for date in unique}
    target = frame["target"].to_numpy()
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    for _ in range(repetitions):
        index = np.concatenate([rows[date] for date in rng.choice(
            unique, len(unique), replace=True)])
        truth = target[index]
        auc.append(roc_auc_score(truth, challenger[index])
                   - roc_auc_score(truth, baseline[index]))
        ap.append(average_precision_score(truth, challenger[index])
                  - average_precision_score(truth, baseline[index]))
    return np.quantile(auc, [.025, .5, .975]), np.quantile(ap, [.025, .5, .975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--options", default=os.path.join(
        ROOT, "data/earnings/option_entry_straddles.parquet"))
    parser.add_argument("--standardized", default=os.path.join(
        ROOT, "data/earnings/option_standardized_terms.parquet"))
    parser.add_argument("--tail", type=float, default=.20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame, temporal_features = build_pre_event_features(pd.read_parquet(args.events))
    frame["target"] = frame["abs_reaction_1d"] >= args.tail
    train = frame[frame["anndats_act"] <= "2022-12-31"].copy()
    validation = frame[(frame["anndats_act"] > "2022-12-31")
                       & (frame["anndats_act"] <= "2023-12-31")].copy()
    test = frame[frame["anndats_act"] > "2023-12-31"].copy()

    graph, sampler, company_map = build_self_event_graph(frame)
    for split in (train, validation, test):
        split["node_id"] = split["permno"].map(company_map)
    tgat = train_model(
        sampler, train, train["node_id"].to_numpy(dtype=np.int64),
        train["target"].to_numpy(), True, args.epochs, 100_000, args.seed)
    validation["tgat"] = predict(
        tgat, validation, validation["node_id"].to_numpy(dtype=np.int64))
    test["tgat"] = predict(tgat, test, test["node_id"].to_numpy(dtype=np.int64))

    options = pd.read_parquet(args.options)
    term = build_standardized_event_variance(pd.read_parquet(args.standardized))
    panels = []
    for split in (train, validation, test):
        panel = build_option_event_panel(split, options).merge(
            term, on="event_id", how="left", validate="one_to_one")
        panel = panel[(panel["implied_move"].between(.005, .80))
                      & panel["option_spread_pct"].between(0, 1.5)].copy()
        panels.append(panel)
    train_o, validation_o, test_o = panels
    option_features = [
        "implied_move", "implied_move_ask", "option_mean_iv", "option_put_skew",
        "option_delta_balance", "option_spread_pct", "option_dte",
        "option_days_after_event", "option_event_sigma", "option_diffusive_variance",
        "std_event_expected_abs", "std_event_sigma", "std_diffusive_variance",
        "std_10d_straddle_pct", "std_10d_iv", "log_call_oi", "log_put_oi",
        "log_call_volume", "log_put_volume",
    ]
    option_features = [column for column in option_features if column in train_o]
    temporal_features = [column for column in temporal_features if column in train_o]

    option_model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=25, learning_rate=.05,
        l2_regularization=3, random_state=args.seed).fit(
            train_o[option_features], train_o["target"])
    scalar_model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=31, learning_rate=.05,
        l2_regularization=2, random_state=args.seed).fit(
            train_o[temporal_features], train_o["target"])
    option_val = option_model.predict_proba(validation_o[option_features])[:, 1]
    option_test = option_model.predict_proba(test_o[option_features])[:, 1]
    scalar_val = scalar_model.predict_proba(validation_o[temporal_features])[:, 1]
    scalar_test = scalar_model.predict_proba(test_o[temporal_features])[:, 1]

    report("options only", test_o["target"], option_test)
    report("TGAT only", test_o["target"], test_o["tgat"])
    candidates = {}
    for name, val_columns, test_columns in (
        ("options + TGAT", [option_val, validation_o["tgat"]],
         [option_test, test_o["tgat"]]),
        ("options + scalar", [option_val, scalar_val], [option_test, scalar_test]),
        ("options + scalar + TGAT",
         [option_val, scalar_val, validation_o["tgat"]],
         [option_test, scalar_test, test_o["tgat"]]),
    ):
        combiner = LogisticRegression(C=1).fit(
            np.column_stack(val_columns), validation_o["target"])
        score = combiner.predict_proba(np.column_stack(test_columns))[:, 1]
        candidates[name] = score
        report(name, test_o["target"], score)
        print(f"  coefficients {combiner.coef_[0]}")
    for name, score in candidates.items():
        auc, ap = date_bootstrap(test_o.reset_index(drop=True), option_test, score,
                                 seed=args.seed)
        print(f"{name} minus options: AUC [{auc[0]:+.4f},{auc[2]:+.4f}] "
              f"median {auc[1]:+.4f} | AP [{ap[0]:+.4f},{ap[2]:+.4f}] "
              f"median {ap[1]:+.4f}")


if __name__ == "__main__":
    main()
