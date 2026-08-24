"""
Supply-chain edges from WRDS Compustat Segment Customer data.

This is the input that makes the graph load-bearing. A co-mention graph says two
firms appeared in the same article; a supplier-customer graph says one firm's
revenue depends on another's demand. Only the second gives a reason for a shock
to propagate, and only a reason makes the propagation worth measuring.

The relationship is temporal -- firms gain and lose customers -- so edges carry
the date the link was reported and the engine's causal sampler answers "who were
A's customers as of time t" rather than "who are they now". Using today's supply
chain to study a 2021 shock is a lookahead bug that no amount of careful return
alignment will save you from.
"""

import glob
import os

import numpy as np
import pandas as pd

# WRDS exports vary by interface and vintage; accept the common spellings
# rather than demanding one exact schema.
SUPPLIER_ID = ("gvkey", "supplier_gvkey", "sgvkey")
CUSTOMER_ID = ("cgvkey", "customer_gvkey", "cid", "cnms")
DATE = ("srcdate", "datadate", "date", "fyear")
SALES = ("salecs", "sales", "cust_sales")
SUPPLIER_TICKER = ("tic", "supplier_tic", "stic")
CUSTOMER_TICKER = ("ctic", "customer_tic")


def _first_present(frame, candidates, required=True, label=""):
    lowered = {c.lower(): c for c in frame.columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    if required:
        raise ValueError(
            f"could not find a {label} column; looked for {candidates}, "
            f"found {sorted(frame.columns)[:12]}")
    return None


def load_supply_chain(path_or_glob, min_sales_share=0.0, quiet=False):
    """
    Load WRDS customer-segment rows into a temporal edge table.

    Args:
        path_or_glob: CSV/parquet file or glob of the WRDS export.
        min_sales_share: Drop links where the customer accounts for less than
            this fraction of the supplier's disclosed segment sales. A customer
            worth 0.3% of revenue is not a channel a shock travels down, and
            keeping it adds noise edges that dilute every neighbourhood.

    Returns:
        DataFrame [ts, src, dst, weight] where src is the supplier, dst the
        customer, and ts the reporting date in unix seconds.
    """
    paths = sorted(glob.glob(path_or_glob)) if not os.path.isfile(path_or_glob) \
        else [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"no supply-chain file matched {path_or_glob!r}")

    frames = []
    for path in paths:
        frames.append(pd.read_parquet(path) if path.endswith(".parquet")
                      else pd.read_csv(path, low_memory=False))
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.lower() for c in raw.columns]

    supplier = _first_present(raw, SUPPLIER_ID, label="supplier id")
    customer = _first_present(raw, CUSTOMER_ID, label="customer id")
    date = _first_present(raw, DATE, label="date")
    sales = _first_present(raw, SALES, required=False)

    table = pd.DataFrame({
        "src": raw[supplier].astype(str).str.strip().str.upper(),
        "dst": raw[customer].astype(str).str.strip().str.upper(),
    })

    # fyear arrives as a bare year; everything else parses as a date.
    if date == "fyear":
        table["ts"] = pd.to_datetime(raw[date].astype("Int64").astype(str) + "-12-31",
                                     errors="coerce")
    else:
        table["ts"] = pd.to_datetime(raw[date], errors="coerce")

    table["weight"] = pd.to_numeric(raw[sales], errors="coerce") if sales else 1.0

    before = len(table)
    table = table.dropna(subset=["ts", "src", "dst"])
    table = table[(table["src"] != "") & (table["dst"] != "")]
    table = table[table["src"] != table["dst"]]

    if min_sales_share > 0 and sales:
        total = table.groupby(["src", table["ts"].dt.year])["weight"].transform("sum")
        table = table[(table["weight"] / total.replace(0, np.nan)) >= min_sales_share]

    table["ts"] = ((table["ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).astype("int64")
    table = table.sort_values("ts", kind="stable").reset_index(drop=True)

    if not quiet:
        span = pd.to_datetime(table["ts"], unit="s")
        print(f"Supply chain: {len(table):,} links ({before - len(table):,} dropped), "
              f"{table['src'].nunique():,} suppliers, {table['dst'].nunique():,} customers")
        if len(table):
            print(f"  span {span.min().date()} -> {span.max().date()}")
    return table[["ts", "src", "dst", "weight"]]


def synthetic_supply_chain(num_firms=300, num_links=2000, seed=0):
    """A supplier graph with the same shape, for testing without WRDS access."""
    rng = np.random.default_rng(seed)
    firms = [f"F{i:04d}" for i in range(num_firms)]

    # Power-law customer concentration: a few hub customers buy from many
    # suppliers, which is what real supply chains look like.
    weights = 1.0 / np.arange(1, num_firms + 1) ** 1.1
    weights /= weights.sum()

    rows = []
    base = pd.Timestamp("2019-01-01").timestamp()
    for _ in range(num_links):
        src = rng.integers(0, num_firms)
        dst = rng.choice(num_firms, p=weights)
        if src == dst:
            continue
        rows.append((int(base + rng.integers(0, 5 * 365) * 86400),
                     firms[src], firms[dst], float(rng.uniform(0.01, 0.4))))
    table = pd.DataFrame(rows, columns=["ts", "src", "dst", "weight"])
    return table.sort_values("ts", kind="stable").reset_index(drop=True)
