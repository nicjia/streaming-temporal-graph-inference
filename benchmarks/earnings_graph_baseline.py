"""Does recent supply-neighbor earnings add to non-graph tail prediction?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from models.earnings_distribution import (build_pre_event_features,
                                          build_supply_event_features,
                                          probability_metrics)
from benchmarks.earnings_distribution_baseline import calibrate, print_metrics


def fit_predict(train, validation, test, features, target):
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=31, learning_rate=.05,
        l2_regularization=2, random_state=0)
    model.fit(train[features], train[target])
    validation_probability = model.predict_proba(validation[features])[:, 1]
    test_probability = model.predict_proba(test[features])[:, 1]
    return calibrate(validation_probability, validation[target], test_probability)


def date_bootstrap(target, baseline, challenger, dates, draws=500, seed=0):
    dates = pd.to_datetime(dates).dt.date.to_numpy()
    unique = np.unique(dates)
    rows = {date: np.flatnonzero(dates == date) for date in unique}
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([rows[date] for date in sampled])
        y = target[index]
        if y.min() == y.max():
            continue
        auc.append(roc_auc_score(y, challenger[index]) - roc_auc_score(y, baseline[index]))
        ap.append(average_precision_score(y, challenger[index])
                  - average_precision_score(y, baseline[index]))
    return np.quantile(auc, [.025, .975]), np.quantile(ap, [.025, .975])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--features", default=os.path.join(
        ROOT, "data/earnings/supply_features.parquet"))
    parser.add_argument("--shuffled", default=os.path.join(
        ROOT, "data/earnings/supply_features_shuffled.parquet"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tail", type=float, default=.15)
    args = parser.parse_args()

    raw = pd.read_parquet(args.data)
    frame, non_graph = build_pre_event_features(raw)
    supply = pd.read_parquet(os.path.join(ROOT, "data/revere_supply_chain.parquet"))
    company_map = pd.read_parquet(os.path.join(ROOT, "data/revere_permno.parquet"))
    if args.refresh or not os.path.exists(args.features):
        graph = build_supply_event_features(frame, supply, company_map)
        graph.to_parquet(args.features, index=False)
    else:
        graph = pd.read_parquet(args.features)
    if args.refresh or not os.path.exists(args.shuffled):
        shuffled = build_supply_event_features(
            frame, supply, company_map, shuffle_neighbors=True, seed=7)
        shuffled.to_parquet(args.shuffled, index=False)
    else:
        shuffled = pd.read_parquet(args.shuffled)
    graph_columns = [column for column in graph if column.startswith("graph_")]
    frame = frame.merge(graph, on="event_id", how="left")
    shuffled = shuffled.rename(columns={column: f"shuffle_{column}"
                                        for column in graph_columns})
    frame = frame.merge(shuffled, on="event_id", how="left")
    shuffled_columns = [f"shuffle_{column}" for column in graph_columns]
    frame["target"] = frame["abs_reaction_1d"] >= args.tail

    train = frame[frame["anndats_act"] <= "2021-12-31"].copy()
    validation = frame[(frame["anndats_act"] > "2021-12-31")
                       & (frame["anndats_act"] <= "2022-12-31")].copy()
    test = frame[frame["anndats_act"] > "2022-12-31"].copy()
    print(f"graph coverage: {(frame['graph_active_neighbors'] > 0).mean():.1%} with active "
          f"neighbors | {(frame['graph_events_90d'] > 0).mean():.1%} with a recent event")

    baseline = fit_predict(train, validation, test, non_graph, "target")
    graph_probability = fit_predict(train, validation, test,
                                    non_graph + graph_columns, "target")
    shuffled_probability = fit_predict(train, validation, test,
                                       non_graph + shuffled_columns, "target")
    graph_only = fit_predict(train, validation, test, graph_columns, "target")
    print_metrics(probability_metrics(test["target"], baseline, "non-graph"))
    print_metrics(probability_metrics(test["target"], graph_only, "supply graph only"))
    print_metrics(probability_metrics(test["target"], shuffled_probability,
                                      "non-graph + shuffled"))
    print_metrics(probability_metrics(test["target"], graph_probability,
                                      "non-graph + supply"))
    auc_ci, ap_ci = date_bootstrap(
        test["target"].to_numpy(), baseline, graph_probability,
        test["anndats_act"], draws=1000)
    print(f"supply minus non-graph date bootstrap: AUC 95% CI "
          f"[{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}] | AP "
          f"[{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
