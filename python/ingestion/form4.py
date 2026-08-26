"""
SEC Form 3/4/5 insider filings as a firm <-> insider bipartite temporal graph.

Free, structured and timestamped: the SEC publishes quarterly TSV bundles of all
ownership filings, so no scraping of half a million individual XML documents.

The bridge is the reporting owner. A director sitting on several boards, or a
fund filing as a 10% holder of several issuers, connects those firms. The
question the graph is built to ask is whether insider activity at one firm
anticipates movement at another firm sharing that person.
"""

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

from ingestion.bipartite import normalise

# Form 4 transaction codes worth keeping: open-market and private purchases and
# sales. Grants, exercises and gifts are compensation mechanics on a vesting
# calendar, not discretionary views, and they swamp the signal if included.
DISCRETIONARY = {"P", "S"}

# RPTOWNER_RELATIONSHIP is a comma-joined label string, e.g. "Director,Officer"
# or "TenPercentOwner", not a numeric code.
def classify_relationship(series):
    lowered = series.astype(str).str.lower()
    return np.select(
        [lowered.str.contains("director"), lowered.str.contains("officer"),
         lowered.str.contains("tenpercent")],
        ["director", "officer", "ten_percent"], default="other")


def relationship_graph(pattern="data/form345/*.zip", roles=("director",),
                       max_issuers_per_bridge=15, quiet=False):
    """
    The board-interlock graph: firm <-> insider, from *every* ownership filing.

    Built from filings rather than from discretionary trades, because the
    relationship is the board seat, not the transaction. A director connects two
    issuers for as long as they sit on both boards whether or not they trade in
    a given quarter, and building edges from trades alone collapses two-hop
    degree from 5 to 1 -- the difference between a usable graph and a dead one.

    `max_issuers_per_bridge` drops institutional 10% filers, which appear on
    hundreds of issuers and are portfolio positions rather than information
    conduits. Density barely moves without them (two-hop median 5 either way),
    which is the evidence that the connectivity is genuine board interlock.
    """
    paths = sorted(glob.glob(pattern))
    frames = []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            if not {"SUBMISSION.tsv", "REPORTINGOWNER.tsv"} <= set(zf.namelist()):
                continue
            sub = _read(zf, "SUBMISSION.tsv",
                        ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK",
                         "ISSUERTRADINGSYMBOL"])
            own = _read(zf, "REPORTINGOWNER.tsv",
                        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
            frames.append(sub.merge(own, on="ACCESSION_NUMBER", how="inner"))

    raw = pd.concat(frames, ignore_index=True)
    filed = pd.to_datetime(raw["FILING_DATE"], errors="coerce", format="mixed")
    raw = raw.assign(_ts=(filed - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))
    raw = raw.dropna(subset=["_ts"])
    raw = raw.assign(_rel=classify_relationship(raw["RPTOWNER_RELATIONSHIP"]))
    if roles:
        raw = raw[raw["_rel"].isin(roles)]

    edges = normalise(raw, ts="_ts", entity="ISSUERCIK", bridge="RPTOWNERCIK",
                      relation="_rel")
    if max_issuers_per_bridge:
        reach = edges.groupby("bridge")["entity"].nunique()
        edges = edges[edges["bridge"].isin(reach[reach <= max_issuers_per_bridge].index)]

    tickers = (raw.dropna(subset=["ISSUERTRADINGSYMBOL"])
                  .assign(ticker=lambda d: d["ISSUERTRADINGSYMBOL"].astype(str).str.strip().str.upper())
                  .groupby(raw["ISSUERCIK"].astype(str))["ticker"].agg(
                      lambda s: s.value_counts().index[0]))
    if not quiet:
        print(f"Interlock graph: {len(edges):,} edges, {edges['entity'].nunique():,} issuers, "
              f"{edges['bridge'].nunique():,} insiders, {len(tickers):,} tickers resolved")
    return edges, tickers


def _read(zf, name, columns=None):
    with zf.open(name) as handle:
        return pd.read_csv(handle, sep="\t", low_memory=False, usecols=columns)


def load_form4(pattern="data/form345/*.zip", discretionary_only=True, quiet=False):
    """
    Returns:
        (edges, meta) where edges follows the bipartite SCHEMA and meta carries
        [ts, issuer, ticker, owner, code, shares, price, value] for event studies.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no Form 345 bundles matched {pattern!r}")

    frames = []
    for path in paths:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if not {"SUBMISSION.tsv", "REPORTINGOWNER.tsv", "NONDERIV_TRANS.tsv"} <= names:
                continue
            sub = _read(zf, "SUBMISSION.tsv",
                        ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK",
                         "ISSUERNAME", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"])
            own = _read(zf, "REPORTINGOWNER.tsv",
                        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                         "RPTOWNER_RELATIONSHIP"])
            trans = _read(zf, "NONDERIV_TRANS.tsv")

            keep = [c for c in ("ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                                "TRANS_SHARES", "TRANS_PRICEPERSHARE",
                                "TRANS_ACQUIRED_DISP_CD") if c in trans.columns]
            trans = trans[keep]

            merged = (trans.merge(sub, on="ACCESSION_NUMBER", how="inner")
                           .merge(own, on="ACCESSION_NUMBER", how="inner"))
            frames.append(merged)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["DOCUMENT_TYPE"].astype(str).str.strip() == "4"]
    if discretionary_only:
        raw = raw[raw["TRANS_CODE"].astype(str).str.strip().isin(DISCRETIONARY)]

    # TRANS_DATE is the trade date; FILING_DATE is when it became public. Only
    # the filing date is tradable information, so that is what the graph uses.
    filed = pd.to_datetime(raw["FILING_DATE"], errors="coerce", format="mixed")
    raw = raw.assign(_ts=(filed - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))
    raw = raw.dropna(subset=["_ts"])

    relationship = raw["RPTOWNER_RELATIONSHIP"].astype(str).str.extract(r"(\d)")[0]
    raw = raw.assign(_rel=relationship.map(RELATIONSHIP).fillna("other"))

    edges = normalise(raw, ts="_ts", entity="ISSUERCIK", bridge="RPTOWNERCIK",
                      relation="_rel", weight="TRANS_SHARES")

    shares = pd.to_numeric(raw.get("TRANS_SHARES"), errors="coerce")
    price = pd.to_numeric(raw.get("TRANS_PRICEPERSHARE"), errors="coerce")
    meta = pd.DataFrame({
        "ts": raw["_ts"].astype("int64").to_numpy(),
        "issuer": raw["ISSUERCIK"].astype(str).to_numpy(),
        "ticker": raw["ISSUERTRADINGSYMBOL"].astype(str).str.strip().str.upper().to_numpy(),
        "owner": raw["RPTOWNERCIK"].astype(str).to_numpy(),
        "relation": raw["_rel"].to_numpy(),
        "code": raw["TRANS_CODE"].astype(str).str.strip().to_numpy(),
        "shares": shares.to_numpy(),
        "price": price.to_numpy(),
    }).sort_values("ts", kind="stable").reset_index(drop=True)
    meta["value"] = meta["shares"] * meta["price"]

    if not quiet:
        span = pd.to_datetime(meta["ts"], unit="s")
        print(f"Form 4: {len(meta):,} discretionary transactions, "
              f"{meta['issuer'].nunique():,} issuers, {meta['owner'].nunique():,} insiders")
        print(f"  span {span.min().date()} -> {span.max().date()} | "
              f"buys {100*(meta['code']=='P').mean():.0f}% / sells {100*(meta['code']=='S').mean():.0f}%")
    return edges, meta
