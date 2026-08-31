"""Decoded DEX swap ingestion and cross-pool markout labels.

The Chainticks sample contains the same USDC/WETH market on Uniswap V2 and
V3.  That makes the *other* pool a contemporaneous, on-chain reference price:
an execution in one pool is marked against the last price reached by the other
pool after ``horizon_blocks``.  Using the other pool avoids calling the swap's
own mechanical price impact information.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
# A dense within-block rank, not the raw Ethereum log index.  Raw log indices
# exceed 6,000 in a few blocks; multiplying those by a safe stride across this
# 548k-block sample would overflow the C++ engine's uint32 timestamp.  At most
# a few dozen swaps from the two focal pools occur in one block, so 4,096 dense
# slots preserve exact order and fit comfortably.
ORDER_SLOTS_PER_BLOCK = 4096


def load_chainticks(root: str | Path) -> pd.DataFrame:
    """Load, validate and chronologically order the public swap partitions."""
    paths = sorted(glob.glob(str(Path(root) / "swaps" / "**" / "*.parquet"),
                             recursive=True))
    if not paths:
        raise FileNotFoundError(f"no Chainticks parquet partitions under {root}")

    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["block"] = pd.to_numeric(frame["block_number"], errors="coerce")
    frame["log_index_int"] = pd.to_numeric(frame["log_index"], errors="coerce")
    frame = frame.dropna(subset=["block", "log_index_int", "sender", "pool_address"])
    frame["block"] = frame["block"].astype(np.int64)
    frame["log_index_int"] = frame["log_index_int"].astype(np.int64)
    frame = (frame.sort_values(["block", "log_index_int"], kind="stable")
             .drop_duplicates(["block", "log_index_int", "pool_address", "tx_hash"])
             .reset_index(drop=True))

    tokens = set(frame["token_in_address"]) | set(frame["token_out_address"])
    if not {WETH, USDC}.issubset(tokens):
        raise ValueError("expected the USDC/WETH Chainticks market")

    is_buy = frame["token_in_address"].eq(USDC).to_numpy()
    is_sell = frame["token_in_address"].eq(WETH).to_numpy()
    if not np.all(is_buy | is_sell):
        raise ValueError("sample contains an unexpected input token")

    amount_in = frame["amount_in"].to_numpy(dtype=np.float64)
    amount_out = frame["amount_out"].to_numpy(dtype=np.float64)
    # One common unit: USDC per WETH, regardless of the swap direction.
    price = np.where(is_buy, amount_in / amount_out, amount_out / amount_in)
    valid = np.isfinite(price) & (price > 0)
    frame = frame.loc[valid].copy().reset_index(drop=True)
    price = price[valid]
    is_buy = is_buy[valid]

    pools = sorted(frame["pool_address"].unique())
    if len(pools) != 2:
        raise ValueError(f"cross-pool benchmark requires two pools, found {len(pools)}")
    pool_map = {pool: index for index, pool in enumerate(pools)}
    frame["pool"] = frame["pool_address"].map(pool_map).astype(np.uint8)
    frame["direction"] = np.where(is_buy, 1, -1).astype(np.int8)
    frame["price"] = price
    frame["log_price"] = np.log(price)
    frame["notional_usdc"] = np.where(
        is_buy,
        frame["amount_in"].to_numpy(dtype=np.float64),
        frame["amount_out"].to_numpy(dtype=np.float64),
    )
    first_block = int(frame["block"].min())
    within_block_rank = frame.groupby("block", sort=False).cumcount().to_numpy() + 1
    if within_block_rank.max(initial=0) >= ORDER_SLOTS_PER_BLOCK:
        raise ValueError("too many focal-pool swaps in one block")
    frame["event_time"] = ((frame["block"].to_numpy() - first_block)
                           * ORDER_SLOTS_PER_BLOCK
                           + within_block_rank).astype(np.int64)
    return frame


def add_cross_pool_markout(swaps: pd.DataFrame, horizon_blocks: int = 3,
                           max_abs_bps: float | None = 500.0) -> pd.DataFrame:
    """Mark each swap against the other pool's close after a fixed horizon.

    Positive ``markout_bps`` means the trader's direction was followed by the
    reference price, hence adverse selection for the pool.  A reference is
    accepted only if the other pool trades after the focal swap and no later
    than the horizon; stale quotes are never carried forward into a label.
    """
    if horizon_blocks < 1:
        raise ValueError("horizon_blocks must be positive")
    out = swaps.copy()
    times = out["event_time"].to_numpy(dtype=np.int64)
    blocks = out["block"].to_numpy(dtype=np.int64)
    pools = out["pool"].to_numpy(dtype=np.int8)
    prices = out["log_price"].to_numpy(dtype=np.float64)
    direction = out["direction"].to_numpy(dtype=np.float64)
    first_block = int(blocks.min())
    markout = np.full(len(out), np.nan, dtype=np.float64)

    for pool in (0, 1):
        focal = np.flatnonzero(pools == pool)
        reference = np.flatnonzero(pools != pool)
        reference_times = times[reference]
        reference_prices = prices[reference]

        horizon_end = ((blocks[focal] - first_block + horizon_blocks + 1)
                       * ORDER_SLOTS_PER_BLOCK - 1)
        last_by_horizon = np.searchsorted(reference_times, horizon_end,
                                          side="right") - 1
        first_after_swap = np.searchsorted(reference_times, times[focal], side="right")
        valid = ((last_by_horizon >= first_after_swap)
                 & (last_by_horizon >= 0))
        rows = focal[valid]
        markout[rows] = (direction[rows]
                         * (reference_prices[last_by_horizon[valid]] - prices[rows])
                         * 10_000.0)

    if max_abs_bps is not None:
        markout = np.clip(markout, -max_abs_bps, max_abs_bps)
    out["markout_bps"] = markout
    out["adverse_selection_usdc"] = (
        out["notional_usdc"].to_numpy(dtype=np.float64) * markout / 10_000.0)
    return out


def pool_fee_bps(frame: pd.DataFrame) -> np.ndarray:
    """Known fee tiers for the two canonical pools in this corpus."""
    version = frame["version"].astype(str).to_numpy()
    # 0x88e6... is the 5 bp V3 pool; canonical Uniswap V2 charges 30 bp.
    return np.where(version == "v3", 5.0, 30.0)
