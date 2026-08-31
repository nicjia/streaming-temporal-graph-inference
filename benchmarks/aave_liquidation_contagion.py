"""Does Aave's collateral/debt graph predict or propagate liquidation risk?

For the first liquidation in each seed block, candidate borrowers are sampled
from the graph state immediately before the event:

* exact: currently supplies the same collateral AND borrows the same debt;
* one-leg: connected to exactly one side of the liquidation pair;
* control: active borrower matched on current portfolio degree but connected
  to neither side.

Forward hazard tests prediction; a symmetric pre-event hazard and their
difference-in-differences distinguish persistent risk concentration from
event-induced propagation. Comparisons are within the same seed event, so
market-wide crashes affect all groups. Standard errors cluster by UTC date.
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.aave import load_aave_events, sample_candidates


def build_candidate_panel(events, liquidations, candidates_per_group=15,
                          exposure_blocks=650_000, seed=0, close_on_exit=True):
    rng = np.random.default_rng(seed)
    complete = liquidations.dropna(subset=["collateral", "debt"]).copy()
    # One seed per block prevents a 100-liquidation block from being treated as
    # 100 independent shocks.  All liquidations remain available as outcomes.
    seeds = (complete.sort_values(["block", "tx_index", "log_index"])
             .drop_duplicates("block", keep="first"))
    seed_at = {(int(row.block), int(row.tx_index), int(row.log_index)): row
               for row in seeds.itertuples(index=False)}

    liquidation_blocks = {
        user: np.sort(group["block"].unique().astype(np.int64))
        for user, group in liquidations.groupby("user")
    }
    supply = defaultdict(set)
    borrow = defaultdict(set)
    user_supply = defaultdict(set)
    user_borrow = defaultdict(set)
    last = {}
    expiry = []
    seen_users, seen_set = [], set()
    records = []

    def remove(kind, asset, user):
        if kind == "supply":
            supply[asset].discard(user); user_supply[user].discard(asset)
        else:
            borrow[asset].discard(user); user_borrow[user].discard(asset)

    def add(kind, asset, user, block):
        key = (kind, asset, user)
        last[key] = block
        if kind == "supply":
            supply[asset].add(user); user_supply[user].add(asset)
        else:
            borrow[asset].add(user); user_borrow[user].add(asset)
        heapq.heappush(expiry, (block + exposure_blocks, kind, asset, user, block))
        if user not in seen_set:
            seen_set.add(user); seen_users.append(user)

    # A single tuple pass is ~20x faster than constructing 1.7M tiny DataFrame
    # groups.  ``seed_at.pop`` guarantees that the collateral/debt copies of a
    # LiquidationCall trigger one snapshot, at the first copy's exact position.
    for row in events.itertuples(index=False):
        block = int(row.block)
        while expiry and expiry[0][0] < block:
            _, kind, asset, user, stamped = heapq.heappop(expiry)
            if last.get((kind, asset, user)) == stamped:
                remove(kind, asset, user)

        seed_row = seed_at.pop((block, int(row.tx_index), int(row.log_index)), None)
        if seed_row is not None:
            collateral, debt, focal = seed_row.collateral, seed_row.debt, seed_row.user
            supplied = supply[collateral]
            borrowed = borrow[debt]
            exact_all = supplied & borrowed
            one_all = (supplied ^ borrowed)
            excluded = {focal}
            exact = sample_candidates(exact_all, candidates_per_group, rng, excluded)
            one = sample_candidates(one_all, candidates_per_group, rng, excluded)

            # Degree-matched controls from neither exposure leg.  Rejection
            # sampling avoids materialising the entire active-user universe at
            # every one of 6k seed blocks.
            reference_degrees = [len(user_supply[u]) + len(user_borrow[u])
                                 for u in exact] or [2]
            target_degree = int(np.median(reference_degrees))
            control = []
            attempts = 0
            while (len(control) < candidates_per_group
                   and attempts < candidates_per_group * 500 and seen_users):
                attempts += 1
                user = seen_users[int(rng.integers(len(seen_users)))]
                degree = len(user_supply[user]) + len(user_borrow[user])
                if (user in excluded or user in exact_all or user in one_all
                        or user in control or degree == 0
                        or abs(degree - target_degree) > 1):
                    continue
                control.append(user)

            for label, users in (("exact", exact), ("one_leg", one),
                                 ("control", control)):
                for user in users:
                    history = liquidation_blocks.get(user, np.empty(0, dtype=np.int64))
                    prior = int(np.searchsorted(history, block, side="left"))
                    previous_block = (int(history[prior - 1]) if prior > 0
                                      else np.iinfo(np.int64).min)
                    next_index = int(np.searchsorted(history, block, side="right"))
                    next_block = (int(history[next_index]) if next_index < len(history)
                                  else np.iinfo(np.int64).max)
                    records.append({
                        "seed_block": block, "seed_time": int(seed_row.timestamp),
                        "group": label, "user": user,
                        "degree": len(user_supply[user]) + len(user_borrow[user]),
                        "prior_liquidations": prior,
                        "previous_block": previous_block,
                        "next_block": next_block,
                        "exact_pool_size": len(exact_all),
                    })

        # Apply the current action only after creating a seed's pre-event
        # snapshot. Liquidation rows themselves do not alter exposure state.
        typ, asset, user = row.event_type, row.asset, row.user
        if not user or not asset:
            continue
        if typ == "supply":
            add("supply", asset, user, block)
        elif typ == "borrow":
            add("borrow", asset, user, block)
        elif typ == "withdraw":
            if close_on_exit:
                remove("supply", asset, user)
        elif typ == "repay":
            if close_on_exit:
                remove("borrow", asset, user)

    return pd.DataFrame(records)


def clustered_difference(panel, horizon, treatment, control="control",
                         first_liquidation_only=False, non_overlapping=False,
                         placebo_before=False, post_minus_pre=False):
    frame = panel[panel["group"].isin([treatment, control])].copy()
    if first_liquidation_only:
        frame = frame[frame["prior_liquidations"] == 0]
    if non_overlapping:
        keep = []
        last = -np.inf
        for block in np.sort(frame["seed_block"].unique()):
            if block > last + horizon:
                keep.append(block); last = block
        frame = frame[frame["seed_block"].isin(keep)]
    if placebo_before or post_minus_pre:
        if "previous_block" not in frame:
            raise ValueError("panel lacks previous_block for the pre-event placebo")
    post = frame["next_block"] <= frame["seed_block"] + horizon
    prior = frame["previous_block"] >= frame["seed_block"] - horizon
    if post_minus_pre:
        frame["hit"] = post.astype(float) - prior.astype(float)
    elif placebo_before:
        frame["hit"] = prior
    else:
        frame["hit"] = post
    per_seed = frame.groupby(["seed_block", "seed_time", "group"])["hit"].mean().unstack()
    per_seed = per_seed.dropna(subset=[treatment, control])

    # Seed fixed effects compare candidates exposed to the exact same market
    # shock. Degree and prior-liquidation controls are demeaned within seed as
    # well. Date-clustered covariance handles overlapping market episodes.
    frame = frame[frame["seed_block"].isin(per_seed.reset_index()["seed_block"])]
    frame["treatment"] = frame["group"].eq(treatment).astype(float)
    frame["log_degree"] = np.log1p(frame["degree"])
    frame["log_prior"] = np.log1p(frame["prior_liquidations"])
    columns = ["hit", "treatment", "log_degree", "log_prior"]
    means = frame.groupby("seed_block")[columns].transform("mean")
    demeaned = frame[columns].astype(float) - means
    dates = pd.to_datetime(frame["seed_time"], unit="s", utc=True).dt.date
    fit = sm.OLS(demeaned["hit"], demeaned[["treatment", "log_degree", "log_prior"]],
                 hasconst=False).fit(cov_type="cluster", cov_kwds={"groups": dates})
    effect = float(fit.params["treatment"])
    t_stat = float(fit.tvalues["treatment"])
    days = len(np.unique(dates))
    raw = frame.groupby("group")["hit"].mean()
    return effect, t_stat, len(per_seed), days, float(raw.get(treatment, np.nan)), float(raw.get(control, np.nan))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "data/amm_events/AaveEventData"))
    parser.add_argument("--candidates", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exposure-blocks", type=int, default=650_000,
                        help="How long a supply/borrow relationship remains active")
    parser.add_argument("--panel-out", default="")
    parser.add_argument("--keep-partial-exposures", action="store_true",
                        help="Do not assume every withdraw/repay closes a position")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    events, liquidations = load_aave_events(args.data, use_cache=not args.refresh)
    panel = build_candidate_panel(events, liquidations, args.candidates,
                                  exposure_blocks=args.exposure_blocks, seed=args.seed,
                                  close_on_exit=not args.keep_partial_exposures)
    if args.panel_out:
        panel.to_parquet(args.panel_out, index=False)
    print(f"Aave: {len(events):,} typed actions | {len(liquidations):,} unique "
          f"liquidations | {liquidations['user'].nunique():,} liquidated borrowers")
    print(f"candidate panel: {len(panel):,} borrower-seed pairs | "
          f"{panel['seed_block'].nunique():,} seed blocks")
    print("\nforward path-conditioned hazard (date-clustered inference)")
    for horizon in (20, 100, 500, 2_000, 7_200):
        minutes = horizon * 12 / 60
        for treatment in ("exact", "one_leg"):
            effect, t, seeds, days, treated, control = clustered_difference(
                panel, horizon, treatment)
            print(f"h={horizon:>4} (~{minutes:>5.0f}m) {treatment:<7} "
                  f"{treated:6.2%} vs {control:6.2%} | lift {effect:+7.3%} "
                  f"t {t:+5.2f} | {seeds:,} seeds/{days} days")

    print("\nrobustness: exact path, first liquidation only / non-overlapping seeds")
    for horizon in (100, 500, 2_000, 7_200):
        first = clustered_difference(panel, horizon, "exact",
                                     first_liquidation_only=True)
        nonoverlap = clustered_difference(panel, horizon, "exact",
                                          non_overlapping=True)
        print(f"h={horizon:>4} first-only lift {first[0]:+7.3%} t {first[1]:+5.2f} | "
              f"non-overlap lift {nonoverlap[0]:+7.3%} t {nonoverlap[1]:+5.2f} "
              f"({nonoverlap[2]:,} seeds)")

    print("\nplacebo: exact-path liquidation hazard before the seed event")
    for horizon in (100, 500, 2_000, 7_200):
        prior = clustered_difference(panel, horizon, "exact", placebo_before=True)
        print(f"h={horizon:>4} prior exact {prior[4]:6.2%} vs control {prior[5]:6.2%} "
              f"| lift {prior[0]:+7.3%} t {prior[1]:+5.2f}")

    print("\nevent-induced effect: (post minus pre) exact path versus control")
    for horizon in (100, 500, 2_000, 7_200):
        change = clustered_difference(panel, horizon, "exact", post_minus_pre=True)
        print(f"h={horizon:>4} exact change {change[4]:+7.3%} vs "
              f"control {change[5]:+7.3%} | DiD {change[0]:+7.3%} "
              f"t {change[1]:+5.2f}")


if __name__ == "__main__":
    main()
