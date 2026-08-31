"""Build the leakage-safe earnings/reaction panel used by options studies."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.earnings import attach_crsp_reactions, download_earnings_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--events", default=os.path.join(
        ROOT, "data/earnings/events.parquet"))
    parser.add_argument("--reactions", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.refresh or not os.path.exists(args.events):
        events = download_earnings_events(args.start, args.end, args.events,
                                          os.path.join(ROOT, ".env"))
    else:
        events = pd.read_parquet(args.events)

    returns = pd.read_parquet(os.path.join(ROOT, "data/crsp_dsf.parquet"),
                              columns=["permno", "date", "ret"])
    extra_path = os.path.join(ROOT, "data/crsp_dsf_extra.parquet")
    if os.path.exists(extra_path):
        extra = pd.read_parquet(extra_path, columns=["permno", "date", "ret"])
        returns = (pd.concat([returns, extra], ignore_index=True)
                   .drop_duplicates(["permno", "date"], keep="first"))
    reactions = attach_crsp_reactions(events, returns)
    os.makedirs(os.path.dirname(args.reactions), exist_ok=True)
    reactions.to_parquet(args.reactions, index=False)
    print(f"events: {len(events):,} across {events['permno'].nunique():,} securities")
    print(f"clean before-open/after-close reactions: {len(reactions):,}")
    print(reactions["announcement_session"].value_counts().to_string())
    for horizon in (1, 2, 5, 10):
        values = reactions[f"reaction_{horizon}d"].abs()
        print(f"|return| h={horizon}: median {values.median():.2%} | "
              f"p90 {values.quantile(.9):.2%} | p99 {values.quantile(.99):.2%}")


if __name__ == "__main__":
    main()
