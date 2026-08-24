"""
Ethereum transaction ingestion.

Benchmark A of the two-benchmark structure: the systems test. It bypasses NLP
and entity extraction entirely and feeds raw on-chain transactions into the C++
engine, because on-chain data *is* a dynamic typed temporal graph rather than a
graph overlaid on something else. Addresses are nodes, transactions are edges,
block time is an exact timestamp, and the call type is a relation code.

It is also the first dataset in this project dense enough for temporal
attention. Measured on 3.16M transactions: the raw graph has median degree 1 --
a long tail of addresses that transact once and never again -- but restricting
to addresses with at least five transactions gives 66,089 nodes at median
degree 6 and mean 35.3, while still covering 64% of all edge endpoints. The
filter is not cherry-picking; it is removing dust.
"""

import glob
import os

import numpy as np
import pandas as pd

# Relation codes. Kept deliberately small and categorical: a learned relation
# embedding needs a vocabulary it can actually see examples of, not a unique
# type per transaction.
REL_UNKNOWN = 0
REL_VALUE_TRANSFER = 1    # plain ETH movement, no calldata
REL_CONTRACT_CALL = 2     # calldata present, non-zero value
REL_CONTRACT_CALL_ZERO = 3  # calldata present, zero value (token transfers, approvals)
REL_CONTRACT_CREATE = 4   # no recipient
REL_FAILED = 5            # reverted

RELATION_NAMES = {
    REL_UNKNOWN: "unknown", REL_VALUE_TRANSFER: "value transfer",
    REL_CONTRACT_CALL: "contract call", REL_CONTRACT_CALL_ZERO: "contract call (0 value)",
    REL_CONTRACT_CREATE: "contract creation", REL_FAILED: "failed",
}

TX_COLUMNS = ["block_number", "from_address", "to_address", "value_f64",
              "gas_used", "success", "n_input_bytes"]


def classify(frame):
    """Assign a relation code per transaction from fields already present."""
    relation = np.full(len(frame), REL_UNKNOWN, dtype=np.uint16)
    has_input = frame["n_input_bytes"].to_numpy() > 0
    has_value = frame["value_f64"].to_numpy() > 0

    relation[~has_input & has_value] = REL_VALUE_TRANSFER
    relation[has_input & has_value] = REL_CONTRACT_CALL
    relation[has_input & ~has_value] = REL_CONTRACT_CALL_ZERO
    relation[frame["to_address"].isna().to_numpy()] = REL_CONTRACT_CREATE
    if "success" in frame:
        relation[~frame["success"].fillna(True).to_numpy().astype(bool)] = REL_FAILED
    return relation


def load_transactions(tx_glob, block_glob, min_degree=5, quiet=False):
    """
    Read transaction and block parquet files into an edge table.

    Args:
        tx_glob, block_glob: globs over the transactions/ and blocks/ files.
        min_degree: drop addresses with fewer than this many transactions. The
            dust tail is 79% of addresses and 36% of endpoints, and it is what
            drags median degree to 1; keeping it would give the temporal
            attention layer nothing to attend over.

    Returns:
        DataFrame [ts, src, dst, relation, value] sorted chronologically, with
        src/dst as dense uint32 ids, plus the id->address mapping.
    """
    tx_paths = sorted(glob.glob(tx_glob))
    block_paths = sorted(glob.glob(block_glob))
    if not tx_paths:
        raise FileNotFoundError(f"no transaction files matched {tx_glob!r}")

    transactions = pd.concat(
        [pd.read_parquet(p, columns=TX_COLUMNS) for p in tx_paths], ignore_index=True)
    blocks = pd.concat([pd.read_parquet(p) for p in block_paths], ignore_index=True)

    time_column = next(c for c in blocks.columns if "time" in c.lower())
    transactions = transactions.merge(
        blocks[["block_number", time_column]], on="block_number", how="left")
    transactions = transactions.rename(columns={time_column: "ts"})
    transactions = transactions.dropna(subset=["from_address", "to_address", "ts"])

    relation = classify(transactions)

    source = transactions["from_address"].to_numpy()
    target = transactions["to_address"].to_numpy()

    # Factorise first, then filter. Doing it the other way round means set
    # membership tests on tens of millions of 20-byte address objects in a
    # Python loop, which is minutes of work; on integer codes it is a
    # vectorised bincount and a take.
    codes, addresses = pd.factorize(pd.Series(np.concatenate([source, target])), sort=False)
    half = len(source)
    src_code = codes[:half]
    dst_code = codes[half:]

    if min_degree > 1:
        degree = np.bincount(codes, minlength=len(addresses))
        active = degree >= min_degree
        keep = active[src_code] & active[dst_code]
        transactions = transactions[keep]
        src_code, dst_code, relation = src_code[keep], dst_code[keep], relation[keep]

        # Re-index so the surviving addresses occupy a dense range; the engine
        # allocates a region per vertex id, so gaps are wasted memory.
        kept = np.zeros(len(addresses), dtype=bool)
        kept[src_code] = True
        kept[dst_code] = True
        remap = np.full(len(addresses), -1, dtype=np.int64)
        surviving = np.flatnonzero(kept)
        remap[surviving] = np.arange(len(surviving))
        src_code, dst_code = remap[src_code], remap[dst_code]
        addresses = addresses[surviving]

    encoded = np.concatenate([src_code, dst_code])
    half = len(src_code)

    table = pd.DataFrame({
        "ts": transactions["ts"].to_numpy().astype("int64"),
        "src": encoded[:half].astype(np.uint32),
        "dst": encoded[half:].astype(np.uint32),
        "relation": relation.astype(np.uint16),
        "value": transactions["value_f64"].to_numpy(),
    }).sort_values("ts", kind="stable").reset_index(drop=True)

    if not quiet:
        span_days = (table["ts"].max() - table["ts"].min()) / 86400
        print(f"Ethereum: {len(table):,} transactions, {len(addresses):,} addresses, "
              f"{span_days:.0f} days of block time")
        share = pd.Series(table["relation"]).map(RELATION_NAMES).value_counts(normalize=True)
        print("  " + " | ".join(f"{k}: {v:.0%}" for k, v in share.items()))

    return table, addresses
