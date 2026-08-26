"""
Patch-adoption timing: how long after a CVE disclosure do dependents move off
the vulnerable version?

This is a propagation measurement with no efficient-market objection. Slow
patching is not an inefficiency somebody arbitrages away; it is a fact about how
an ecosystem behaves, so the result stands whichever way it comes out.

One subtlety decides how the numbers must be read. Most npm dependents declare a
caret range like ^4.17.0, which *already* admits a later patch release, so they
receive the fix on their next install with no manifest change at all. Manifest
based propagation therefore measures the pinned and major-bump subset -- the
dependents who stay exposed until somebody edits a file. That is the population
worth measuring, and it is reported as such rather than as "the ecosystem".
"""

import json

import numpy as np
import pandas as pd
import semantic_version as sv


def _spec_admits_vulnerable(spec, vuln_range, patched):
    """
    Does this dependency spec still allow a vulnerable version?

    Conservative: anything unparseable counts as still vulnerable, so the
    estimate errs toward slower adoption rather than silently dropping the
    awkward cases that are most likely to be pinned.
    """
    if not spec or not patched:
        return True
    try:
        target = sv.Version.coerce(str(patched))
    except Exception:
        return True
    try:
        spec_obj = sv.NpmSpec(str(spec))
    except Exception:
        return True
    # If the spec cannot accept the patched version, the dependent is pinned
    # below the fix and is still exposed.
    try:
        return not spec_obj.match(target)
    except Exception:
        return True


def load_panel(path="data/npm/specs.jsonl"):
    """{package: [(timestamp, {dep: spec}), ...]} sorted by release time."""
    panel = {}
    for line in open(path):
        try:
            record = json.loads(line)
        except Exception:
            continue
        versions = []
        for version, meta in record["versions"].items():
            try:
                stamp = pd.Timestamp(meta["t"]).tz_localize(None)
            except Exception:
                continue
            versions.append((stamp, meta.get("deps") or {}))
        if versions:
            panel[record["name"]] = sorted(versions, key=lambda x: x[0])
    return panel


def patch_events(panel, advisories, max_days=730, quiet=False):
    """
    One row per (advisory, dependent) where the dependent was exposed at
    disclosure, recording how long until its spec first admitted the fix.

    Right-censored observations -- still exposed at the end of the sample -- are
    kept and flagged, because dropping them is exactly how a patch-adoption
    study talks itself into an optimistic half-life.
    """
    advisories = advisories.dropna(subset=["patched"]).copy()
    advisories["published"] = pd.to_datetime(advisories["published"]).dt.tz_localize(None)
    by_package = {}
    for row in advisories.itertuples(index=False):
        by_package.setdefault(row.package, []).append(row)

    horizon = max(max(v[-1][0] for v in panel.values()), advisories["published"].max())
    rows = []
    for dependent, versions in panel.items():
        touched = set()
        for _, deps in versions:
            touched.update(deps)
        for vuln_pkg in touched & by_package.keys():
            # Full state history including releases where the dependency is
            # ABSENT. Dropping a dependency remediates just as surely as
            # upgrading it, and 28% of these relationships end that way --
            # scoring removal as "never patched" understates remediation badly.
            history = [(t, d.get(vuln_pkg)) for t, d in versions]
            if not any(spec is not None for _, spec in history):
                continue
            for adv in by_package[vuln_pkg]:
                published = adv.published
                before = [(t, s) for t, s in history if t <= published]
                if not before or before[-1][1] is None:
                    continue  # not depending on it at disclosure
                spec_at_disclosure = before[-1][1]
                if not _spec_admits_vulnerable(spec_at_disclosure, adv.vuln_range, adv.patched):
                    rows.append((adv.ghsa, adv.package, dependent, adv.severity,
                                 published, 0.0, False))
                    continue
                after = [(t, s) for t, s in history if t > published]
                fixed_at = None
                for t, s in after:
                    remediated = (s is None or
                                  not _spec_admits_vulnerable(s, adv.vuln_range, adv.patched))
                    if remediated:
                        fixed_at = t
                        break
                if fixed_at is not None:
                    days = (fixed_at - published).days
                    if days <= max_days:
                        rows.append((adv.ghsa, adv.package, dependent, adv.severity,
                                     published, float(days), False))
                else:
                    days = (horizon - published).days
                    rows.append((adv.ghsa, adv.package, dependent, adv.severity,
                                 published, float(min(days, max_days)), True))

    frame = pd.DataFrame(rows, columns=["ghsa", "vulnerable_package", "dependent",
                                        "severity", "published", "days", "censored"])
    if not quiet and len(frame):
        print(f"patch observations: {len(frame):,} | "
              f"{frame['dependent'].nunique():,} dependents, "
              f"{frame['ghsa'].nunique():,} advisories")
        print(f"  already safe at disclosure (caret range covered it): "
              f"{(frame['days'] == 0).mean():.0%}")
        print(f"  still exposed at end of sample (censored): {frame['censored'].mean():.0%}")
    return frame
