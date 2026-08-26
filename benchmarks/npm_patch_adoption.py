"""
How fast does a CVE fix propagate through the npm dependency graph?

A propagation measurement with no efficient-market objection: slow patching is
not an inefficiency anyone competes away, so the result stands whichever way it
comes out. That is the structural reason this domain yields findings where five
successive finance datasets did not.

Scope, stated precisely because it decides how to read the numbers. Most npm
dependents declare a caret range like ^4.17.0, which already admits a later
patch release, so they receive the fix on their next install with no manifest
change at all. Those are excluded: they were never exposed. What remains is the
population whose declared spec *cannot* accept the fix -- pinned versions and
fixes that need a major bump -- who stay vulnerable until a human edits a file.

Two corrections that changed the answer materially:

  removal is remediation. Dropping a dependency fixes the exposure as surely as
  upgrading it, and 28% of these relationships end that way. Scoring removal as
  "never patched" put the remediation rate at 7.9%; counting it puts it at 36%.

  abandonment is not slow patching. 58% of exposed dependents never ship another
  release at all and structurally cannot remediate. They are reported separately
  rather than dragged into the survival curve.

Usage:  python benchmarks/npm_patch_adoption.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.npm_cve import load_panel, patch_events  # noqa: E402


def last_release(path):
    latest = {}
    for line in open(path):
        try:
            record = json.loads(line)
        except Exception:
            continue
        stamps = [pd.Timestamp(m["t"]).tz_localize(None)
                  for m in record["versions"].values() if m.get("t")]
        if stamps:
            latest[record["name"]] = max(stamps)
    return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-followup", type=int, default=365)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    events_path = os.path.join(ROOT, "data/npm/patch_events.parquet")
    if args.rebuild or not os.path.exists(events_path):
        advisories = pd.read_parquet(os.path.join(ROOT, "data/npm/advisories.parquet"))
        frame = patch_events(load_panel(os.path.join(ROOT, "data/npm/specs.jsonl")),
                             advisories)
        frame.to_parquet(events_path, index=False)
    events = pd.read_parquet(events_path)
    events["published"] = pd.to_datetime(events["published"])

    latest = last_release(os.path.join(ROOT, "data/npm/specs.jsonl"))
    horizon = max(latest.values())
    events["maintained"] = events["dependent"].map(latest) > events["published"]

    # "Exposed" means the spec at disclosure could not accept the fix.
    exposed = events[~((events["days"] == 0) & (~events["censored"]))]
    print(f"observations {len(events):,} | exposed at disclosure {len(exposed):,} "
          f"({1 - len(exposed)/len(events):.0%} were already covered by a caret range)")
    print(f"  of exposed, still shipping releases: {exposed['maintained'].mean():.0%}")

    live = exposed[exposed["maintained"] &
                   ((horizon - exposed["published"]).dt.days >= args.min_followup)]
    fixed = live[~live["censored"]]
    print(f"\nmaintained dependents with >={args.min_followup}d follow-up: "
          f"{len(live):,} observations, {live['dependent'].nunique():,} packages, "
          f"{live['ghsa'].nunique():,} advisories")
    print(f"  remediated: {len(fixed)/len(live):.1%}")
    if len(fixed):
        q = fixed["days"].quantile([.25, .5, .75])
        print(f"  days to remediate, conditional on remediating: "
              f"p25 {q[.25]:.0f}  median {q[.5]:.0f}  p75 {q[.75]:.0f}")

    print(f"\n  {'day':<8}{'still exposed':>15}")
    print("  " + "-" * 23)
    for day in (7, 30, 90, 180, 365):
        print(f"  {day:<8}{1 - (fixed['days'] <= day).sum()/len(live):>14.0%}")
    half = next((d for d in range(1, 366)
                 if (fixed["days"] <= d).sum()/len(live) >= 0.5), None)
    print(f"  half-life: {str(half) + ' days' if half else '>365 days (never reaches 50%)'}")

    print(f"\n  {'severity':<10}{'n':>6}{'remediated':>13}{'median days':>13}")
    print("  " + "-" * 42)
    for severity in ("critical", "high", "medium", "low"):
        group = live[live["severity"] == severity]
        if len(group) < 30:
            continue
        done = group[~group["censored"]]
        median = done["days"].median() if len(done) else float("nan")
        print(f"  {severity:<10}{len(group):>6}{len(done)/len(group):>12.1%}{median:>13.0f}")

    stranded = exposed[~exposed["maintained"]]
    print(f"\nunmaintained: {len(stranded):,} observations across "
          f"{stranded['dependent'].nunique():,} packages that never shipped again "
          f"and cannot remediate")


if __name__ == "__main__":
    main()
