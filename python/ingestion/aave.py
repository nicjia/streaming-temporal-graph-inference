"""Aave event archive ingestion for borrower/asset temporal graphs."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sample_candidates(values, n: int, rng: np.random.Generator, exclude):
    """Sample a set reproducibly under a declared NumPy seed."""
    # Set iteration is hash-randomized between Python processes. Sorting first
    # makes the declared RNG seed sufficient to reproduce the candidate panel.
    ordered = sorted(values - exclude)
    if not ordered:
        return []
    if len(ordered) <= n:
        return ordered
    return [ordered[i] for i in rng.choice(len(ordered), n, replace=False)]


def load_aave_events(root: str | Path, use_cache: bool = True):
    """Return deduplicated protocol actions and liquidation pairs.

    LiquidationCall is indexed once from its collateral file and once from its
    debt file.  The action table intentionally retains the two typed user-asset
    edges, while the liquidation table merges them into one economic event.
    """
    root = Path(root)
    cache_dir = root.parent / "prepared"
    event_cache = cache_dir / "events.parquet"
    liquidation_cache = cache_dir / "liquidations.parquet"
    if use_cache and event_cache.exists() and liquidation_cache.exists():
        return pd.read_parquet(event_cache), pd.read_parquet(liquidation_cache)

    paths = sorted(glob.glob(str(root / "*.json")))
    if not paths:
        raise FileNotFoundError(f"no Aave event JSON files under {root}")

    frames = []
    liquidations = {}
    for path in paths:
        with open(path) as handle:
            raw = json.load(handle)
        rows = []
        for event in raw:
            event_type = event["event_type"]
            user = str(event.get("user", "")).lower()
            asset = str(event.get("asset", "")).lower()
            role = str(event.get("asset_role", "reserve")).lower()
            rows.append((event_type, int(event["block_number"]),
                         int(event["timestamp"]), int(event.get("tx_index", 0)),
                         int(event.get("log_index", 0)), event["transaction_hash"],
                         user, asset, role))
            if event_type == "liquidationCall":
                key = (event["transaction_hash"], int(event.get("log_index", 0)))
                item = liquidations.setdefault(key, {
                    "block": int(event["block_number"]),
                    "timestamp": int(event["timestamp"]),
                    "tx_index": int(event.get("tx_index", 0)),
                    "log_index": int(event.get("log_index", 0)),
                    "transaction_hash": event["transaction_hash"],
                    "user": user, "collateral": None, "debt": None,
                })
                if role == "collateral":
                    item["collateral"] = asset
                    debt = str(event.get("debt_asset", "")).lower()
                    item["debt"] = debt or item["debt"]
                elif role == "debt":
                    item["debt"] = asset
        frames.append(pd.DataFrame(rows, columns=[
            "event_type", "block", "timestamp", "tx_index", "log_index",
            "transaction_hash", "user", "asset", "asset_role",
        ]))

    events = pd.concat(frames, ignore_index=True)
    events = (events.drop_duplicates([
        "event_type", "transaction_hash", "log_index", "user", "asset", "asset_role"])
        .sort_values(["block", "tx_index", "log_index", "asset"], kind="stable")
        .reset_index(drop=True))
    liquidation_frame = (pd.DataFrame(liquidations.values())
                         .sort_values(["block", "tx_index", "log_index"], kind="stable")
                         .reset_index(drop=True))
    cache_dir.mkdir(parents=True, exist_ok=True)
    events.to_parquet(event_cache, index=False)
    liquidation_frame.to_parquet(liquidation_cache, index=False)
    return events, liquidation_frame
