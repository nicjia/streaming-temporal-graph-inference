"""Block-ahead DEX adverse-selection benchmark.

Decision time is the end of block b.  The target is whether swaps arriving in
pool p during block b+1 are in the most toxic training-decile when marked to
the *other* pool three blocks later.  No transaction, sender or direction from
the target block is visible to any model.

The economic diagnostic is an always-on LP versus a policy that withdraws for
the 10% of holdout blocks with the highest predicted toxicity.  Dollar values
are aggregate pool values before multiplying by an LP's pro-rata share; both
fees and adverse selection scale by that same share.  Gas and endogenous price
impact are deliberately not claimed, so this is a screening test rather than
a deployable backtest.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.dex import add_cross_pool_markout, load_chainticks, pool_fee_bps


def _last(values: pd.Series) -> float:
    return float(values.iloc[-1])


def build_block_panel(swaps: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Causal block-close features and next-block labels for each pool."""
    marked = swaps.dropna(subset=["markout_bps"]).copy()
    marked["signed_notional"] = (marked["direction"].to_numpy()
                                      * marked["notional_usdc"].to_numpy())
    marked["fee_usdc"] = (marked["notional_usdc"].to_numpy()
                           * pool_fee_bps(marked) / 10_000.0)
    marked["toxic_usdc"] = np.maximum(marked["adverse_selection_usdc"], 0.0)

    # A switch is a time-respecting pool -> actor -> other-pool motif.  The
    # previous interaction is found before grouping, so it cannot use future
    # behavior or the next block's actor identities.
    prior_pool = marked.groupby("sender", sort=False)["pool"].shift(1)
    prior_block = marked.groupby("sender", sort=False)["block"].shift(1)
    marked["recent_cross_pool_actor"] = (
        prior_pool.notna()
        & prior_pool.ne(marked["pool"])
        & (marked["block"] - prior_block <= 20)
    ).astype(np.int8)
    marked["multi_pool_tx"] = marked.groupby("tx_hash")["pool"].transform("nunique").gt(1).astype(np.int8)

    block = marked.groupby(["block", "pool"], sort=True).agg(
        count=("pool", "size"),
        volume=("notional_usdc", "sum"),
        signed_volume=("signed_notional", "sum"),
        close=("log_price", _last),
        cross_actor=("recent_cross_pool_actor", "sum"),
        multi_pool=("multi_pool_tx", "sum"),
        markout=("markout_bps", lambda x: float(np.average(
            x, weights=marked.loc[x.index, "notional_usdc"]))),
        adverse=("adverse_selection_usdc", "sum"),
        toxic=("toxic_usdc", "sum"),
        fees=("fee_usdc", "sum"),
    ).reset_index()

    first, last = int(marked["block"].min()), int(marked["block"].max())
    grid = pd.MultiIndex.from_product([np.arange(first, last + 1), [0, 1]],
                                      names=["block", "pool"])
    panel = block.set_index(["block", "pool"]).reindex(grid).reset_index()
    for name in ["count", "volume", "signed_volume", "cross_actor", "multi_pool",
                 "adverse", "toxic", "fees"]:
        panel[name] = panel[name].fillna(0.0)

    panel = panel.sort_values(["pool", "block"], kind="stable")
    group = panel.groupby("pool", sort=False)
    panel["close"] = group["close"].ffill()
    panel["return_1"] = group["close"].diff().fillna(0.0)
    for window in (1, 5, 20, 100):
        for name in ("count", "volume", "signed_volume", "cross_actor", "multi_pool"):
            panel[f"{name}_{window}"] = group[name].transform(
                lambda x, w=window: x.rolling(w, min_periods=1).sum())
    panel["own_close"] = panel["close"]
    other = panel[["block", "pool", "close"]].copy()
    other["pool"] = 1 - other["pool"]
    other = other.rename(columns={"close": "other_close"})
    panel = panel.merge(other, on=["block", "pool"], how="left")
    panel["price_gap_bps"] = (panel["own_close"] - panel["other_close"]) * 10_000.0

    # Shift within venue: all targets occur in block b+1.  Markout itself is
    # realized later, but no part of it enters the block-b feature matrix.
    group = panel.groupby("pool", sort=False)
    for name in ("count", "volume", "markout", "adverse", "toxic", "fees"):
        panel[f"next_{name}"] = group[name].shift(-1)
    panel["next_lp_pnl"] = panel["next_fees"] - panel["next_adverse"]

    features = ["return_1", "price_gap_bps"]
    for window in (1, 5, 20, 100):
        features.extend([f"{name}_{window}" for name in
                         ("count", "volume", "signed_volume", "cross_actor", "multi_pool")])
    panel[features] = panel[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    panel = panel.dropna(subset=["next_count"]).reset_index(drop=True)
    return panel, features


def report_model(name: str, probability: np.ndarray, target: np.ndarray,
                 pnl: np.ndarray, fees: np.ndarray, adverse: np.ndarray) -> dict:
    auc = roc_auc_score(target, probability)
    ap = average_precision_score(target, probability)
    cutoff = np.quantile(probability, 0.90)
    avoid = probability >= cutoff
    always = float(np.sum(pnl))
    gated = float(np.sum(pnl[~avoid]))
    fee_kept = float(np.sum(fees[~avoid]) / max(np.sum(fees), 1e-12))
    toxic_avoided = float(np.sum(adverse[avoid]) / max(np.sum(adverse), 1e-12))
    print(f"{name:<24} AUC {auc:.3f} | AP {ap:.3f} | "
          f"toxic avoided {toxic_avoided:5.1%} | fees kept {fee_kept:5.1%} | "
          f"LP diagnostic {always:,.0f} -> {gated:,.0f}")
    return {"auc": auc, "ap": ap, "toxic_avoided": toxic_avoided,
            "fee_kept": fee_kept, "always_pnl": always, "gated_pnl": gated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "data/chainticks"))
    parser.add_argument("--markout-blocks", type=int, default=3)
    args = parser.parse_args()

    swaps = add_cross_pool_markout(load_chainticks(args.data), args.markout_blocks)
    panel, features = build_block_panel(swaps)
    # The denser V3 pool is the investable venue; V2 histories remain in the
    # price-gap and cross-pool motif features.
    sample = panel[panel["pool"] == 0].sort_values("block").reset_index(drop=True)
    cut_train = int(len(sample) * 0.60)
    cut_val = int(len(sample) * 0.80)
    train, test = sample.iloc[:cut_train], sample.iloc[cut_val:]

    # High toxicity is defined once on training data, in aggregate dollars.
    # Zeros remain legitimate quiet blocks rather than being discarded.
    threshold = float(train["next_toxic"].quantile(0.90))
    y_train = (train["next_toxic"].to_numpy() > threshold).astype(np.int8)
    y_test = (test["next_toxic"].to_numpy() > threshold).astype(np.int8)
    X_train, X_test = train[features].to_numpy(), test[features].to_numpy()

    print(f"DEX: {len(swaps):,} swaps | {swaps['sender'].nunique():,} actors | "
          f"{swaps['block'].nunique():,} active blocks")
    print(f"V3 decision blocks: {len(sample):,} | train {len(train):,} | "
          f"holdout {len(test):,} | toxic prevalence {y_test.mean():.1%}")
    print(f"training 90th-percentile toxic-dollar threshold: ${threshold:,.2f}\n")

    models = {
        "history logistic": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, C=0.2)),
        "history nonlinear": HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=15, learning_rate=0.05,
            l2_regularization=1.0, random_state=0),
    }
    pnl = test["next_lp_pnl"].fillna(0).to_numpy()
    fees = test["next_fees"].fillna(0).to_numpy()
    adverse = test["next_toxic"].fillna(0).to_numpy()
    for name, model in models.items():
        model.fit(X_train, y_train)
        report_model(name, model.predict_proba(X_test)[:, 1], y_test,
                     pnl, fees, adverse)

    # Explicit graph-feature ablation: actor-switch and multi-pool route motifs
    # are the only features unavailable from a univariate pool history.
    plain = [name for name in features
             if "cross_actor" not in name and "multi_pool" not in name]
    model = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, learning_rate=0.05,
        l2_regularization=1.0, random_state=0)
    model.fit(train[plain], y_train)
    report_model("no graph motifs", model.predict_proba(test[plain])[:, 1],
                 y_test, pnl, fees, adverse)


if __name__ == "__main__":
    main()

