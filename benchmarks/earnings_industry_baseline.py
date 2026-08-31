"""Do recent dated industry-peer earnings identify move-tail regimes?"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.earnings import (attach_dated_industry,
                                download_crsp_industry_history)
from models.earnings_distribution import (build_industry_event_features,
                                          build_pre_event_features,
                                          probability_metrics)
from benchmarks.earnings_distribution_baseline import print_metrics
from benchmarks.earnings_graph_baseline import date_bootstrap, fit_predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--names", default=os.path.join(
        ROOT, "data/earnings/crsp_industry_history.parquet"))
    parser.add_argument("--features", default=os.path.join(
        ROOT, "data/earnings/industry_features.parquet"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tail", type=float, default=.15)
    args = parser.parse_args()

    raw = pd.read_parquet(args.data)
    frame, non_graph = build_pre_event_features(raw)
    if not os.path.exists(args.names):
        names = download_crsp_industry_history(
            args.names, os.path.join(ROOT, ".env"))
    else:
        names = pd.read_parquet(args.names)
    frame = attach_dated_industry(frame, names)
    if args.refresh or not os.path.exists(args.features):
        industry = build_industry_event_features(frame)
        industry.to_parquet(args.features, index=False)
    else:
        industry = pd.read_parquet(args.features)
    industry_columns = [column for column in industry if column.startswith("industry_")]
    frame = frame.merge(industry, on="event_id", how="left")
    frame["target"] = frame["abs_reaction_1d"] >= args.tail

    train = frame[frame["anndats_act"] <= "2021-12-31"].copy()
    validation = frame[(frame["anndats_act"] > "2021-12-31")
                       & (frame["anndats_act"] <= "2022-12-31")].copy()
    test = frame[frame["anndats_act"] > "2022-12-31"].copy()
    baseline = fit_predict(train, validation, test, non_graph, "target")
    industry_probability = fit_predict(
        train, validation, test, non_graph + industry_columns, "target")
    industry_only = fit_predict(train, validation, test, industry_columns, "target")
    print(f"dated classifications: SIC2 {frame['sic2'].notna().mean():.1%} | "
          f"NAICS3 {frame['naics3'].notna().mean():.1%}")
    print_metrics(probability_metrics(test["target"], baseline, "non-graph"))
    print_metrics(probability_metrics(test["target"], industry_only, "industry graph only"))
    print_metrics(probability_metrics(test["target"], industry_probability,
                                      "non-graph + industry"))
    auc_ci, ap_ci = date_bootstrap(
        test["target"].to_numpy(), baseline, industry_probability,
        test["anndats_act"], draws=1000)
    print(f"industry minus non-graph date bootstrap: AUC 95% CI "
          f"[{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}] | AP "
          f"[{ap_ci[0]:+.4f}, {ap_ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
