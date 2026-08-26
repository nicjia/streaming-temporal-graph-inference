"""
I/B/E/S analyst coverage as a firm <-> analyst bipartite temporal graph.

The densest graph in this project by an order of magnitude: 11,180 analysts
bridging 8,331 firms, two-hop degree of 58 peers per firm per quarter against
1.7 for supplier links and 1.9 for earnings-call mentions.

The bridge is the analyst. When one analyst revises an estimate, the question is
whether the other firms that same analyst covers move -- information travelling
firm -> analyst -> firm. Collapsing that into a firm-firm edge throws away which
analyst connected them and when they last touched each side, which is the
mechanism itself.

As with board interlocks, the edge is the *coverage relationship*, not the
revision. An analyst covers a firm continuously between revisions, and building
edges only from revision events understates connectivity badly.
"""

import numpy as np
import pandas as pd

from ingestion.bipartite import normalise


def link_to_crsp(revisions, idsum, names, quiet=False):
    """
    Map IBES tickers to CRSP permnos through 8-digit CUSIP.

    IBES and CRSP tickers disagree often enough that linking on ticker is a
    silent source of mismatched firms; CUSIP is the reliable join, and
    dsenames carries the date ranges over which each mapping is valid.
    """
    cusip = revisions[["ticker", "cusip"]].dropna().drop_duplicates()
    cusip["cusip"] = cusip["cusip"].astype(str).str.strip().str.upper()

    valid = names.dropna(subset=["ncusip"]).copy()
    valid["ncusip"] = valid["ncusip"].astype(str).str.strip().str.upper()
    best = (valid.sort_values("namedt")
                 .drop_duplicates("ncusip", keep="last")[["ncusip", "permno", "crsp_ticker"]])

    mapping = cusip.merge(best, left_on="cusip", right_on="ncusip", how="inner")
    mapping = mapping.drop_duplicates("ticker")[["ticker", "permno", "crsp_ticker"]]
    if not quiet:
        print(f"IBES->CRSP link: {len(mapping):,} of {cusip['ticker'].nunique():,} "
              f"IBES tickers matched to a permno")
    return mapping


def revision_events(revisions, min_abs_pct=0.02, top_quantile=0.9, quiet=False):
    """
    Signed estimate revisions, keeping the large ones.

    A revision is measured against that analyst's own previous estimate for the
    same firm and fiscal period, so it captures a change of view rather than a
    level difference between optimistic and pessimistic analysts. Sign orients
    upgrades and downgrades so they reinforce.
    """
    frame = revisions.dropna(subset=["value", "anndats", "analys", "ticker"]).copy()
    frame["date"] = pd.to_datetime(frame["anndats"])
    frame = frame.sort_values("date")

    key = ["analys", "ticker", "fpedats", "fpi"]
    frame["prev"] = frame.groupby(key)["value"].shift(1)
    frame = frame.dropna(subset=["prev"])
    frame = frame[frame["prev"].abs() > 1e-6]

    frame["change"] = (frame["value"] - frame["prev"]) / frame["prev"].abs()
    frame = frame[frame["change"].abs() >= min_abs_pct]

    cut = frame["change"].abs().quantile(top_quantile)
    events = frame[frame["change"].abs() >= cut].copy()
    events["sign"] = np.sign(events["change"])

    if not quiet:
        print(f"Revision events: {len(events):,} "
              f"({(events['sign'] > 0).mean():.0%} upgrades), "
              f"{events['ticker'].nunique():,} firms, {events['analys'].nunique():,} analysts")
    return events[["date", "ticker", "analys", "change", "sign"]].reset_index(drop=True)


def coverage_edges(revisions, quiet=False):
    """The firm <-> analyst coverage graph, one edge per analyst-firm-quarter."""
    frame = revisions.dropna(subset=["analys", "ticker", "anndats"]).copy()
    frame["date"] = pd.to_datetime(frame["anndats"])
    frame["quarter"] = frame["date"].dt.to_period("Q").dt.start_time
    unique = frame.drop_duplicates(["analys", "ticker", "quarter"])
    unique = unique.assign(
        _ts=(unique["quarter"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))
    edges = normalise(unique, ts="_ts", entity="ticker", bridge="analys")
    if not quiet:
        print(f"Coverage graph: {len(edges):,} firm-analyst-quarters, "
              f"{edges['entity'].nunique():,} firms, {edges['bridge'].nunique():,} analysts")
    return edges
