"""
Combine multiple edge sources into one temporal firm graph.

Each source is reduced to the same tuple before merging:

    (src_ticker, dst_ticker, unix_ts, source, relation, weight)

The source id is packed alongside the relation code so provenance survives the
merge and each source's contribution can be compared. Three sources are wired in,
all keyed on ticker:
  text_8k       firm->firm supplier/competitor/partner/owner/lender, from 8-K text
  interlock     firm<->firm sharing a director, from SEC Form 3/4/5 filings
  supply_chain  supplier->customer, from FactSet Revere

Usage:
    python benchmarks/unified_graph.py --edges-8k 'data/edgar/edges/targeted/*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.entity_resolve import EntityResolver  # noqa: E402
from ingestion.extraction_schema import STRUCTURAL, Relation  # noqa: E402

# Source ids, packed into the high bits of the uint16 relation channel so a
# merged edge still says where it came from. Relation uses bits 0-7 and
# confidence 8-9; source takes bits 11-13, leaving room for eight sources.
SOURCES = {"text_8k": 0, "interlock": 1, "supply_chain": 2, "comovement": 3}
_SOURCE_SHIFT = 11

# Interlock is a single relation type of its own, distinct from the 8-K enum.
REL_INTERLOCK = 20


def _pack(relation: int, source: str) -> int:
    return (relation & 0xFF) | ((SOURCES[source] & 0x7) << _SOURCE_SHIFT)


def _ts(date_str: str) -> int:
    return int(datetime.strptime(date_str[:10], "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp())


def load_8k(pattern: str, resolver: EntityResolver):
    """8-K text edges, resolved to tickers, structural relations only.

    Event relations (earnings, leadership) are dropped: they are mostly
    self-loops on the filer and add no cross-firm structure.
    """
    edges = []
    paths = [q for pat in pattern.split(",") for q in sorted(glob.glob(pat.strip()))]
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stamp = _ts(rec["date_filed"])
            filer = rec.get("ticker")
            for e in rec["edges"]:
                rel = Relation[e["relation"]]
                if rel not in STRUCTURAL:
                    continue
                s = resolver.resolve(e["subject"], filer)
                d = resolver.resolve(e["object"], filer)
                if not s or not d or s == d:
                    continue
                if s[0] != "anchored" or d[0] != "anchored":
                    continue  # tradeable subgraph: listed <-> listed
                edges.append((s[1], d[1], stamp, "text_8k",
                              _pack(int(rel), "text_8k"), float(e["weight"])))
    return edges


def load_interlock(pattern: str, listed_tickers: set, quiet: bool = True):
    """Firm<->firm edges from shared directors in SEC ownership filings.

    The form4 loader gives a firm<->insider bipartite graph; two firms are
    connected if the same director filed for both in the window, timestamped at
    the later filing. Issuers arrive keyed by ticker and map to the vertex space
    directly, not through the name resolver used on the 8-K side.
    """
    from ingestion.form4 import relationship_graph

    bip, tickers = relationship_graph(pattern=pattern, roles=("director",),
                                      max_issuers_per_bridge=15, quiet=quiet)
    cik_to_ticker = {str(k): v for k, v in tickers.items()}

    by_bridge = defaultdict(list)
    for row in bip.itertuples(index=False):
        tkr = cik_to_ticker.get(str(row.entity))
        if not tkr or tkr not in listed_tickers:
            continue
        by_bridge[row.bridge].append((tkr, int(row.ts)))

    edges = []
    seen = set()
    for bridge, members in by_bridge.items():
        # A director on k boards makes k*(k-1)/2 firm pairs; cap to avoid a
        # single hub creating a clique, mirroring the bipartite reach cap.
        uniq = {}
        for node, ts in members:
            uniq[node] = max(uniq.get(node, 0), ts)
        firms = sorted(uniq)
        if len(firms) > 15:
            continue
        for i in range(len(firms)):
            for j in range(i + 1, len(firms)):
                a, b = firms[i], firms[j]
                ts = max(uniq[a], uniq[b])
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                edges.append((a, b, ts, "interlock",
                              _pack(REL_INTERLOCK, "interlock"), 0.0))
    return edges


REL_SUPPLY = 21


def load_supply_chain(listed_tickers: set, window=("2020-01-01", "2025-12-31"),
                      quiet: bool = True):
    """Supplier->customer edges from FactSet Revere, mapped to ticker.

    Revere keys companies by its own id; the chain to the vertex space is
    company_id -> permno -> ticker. Only relationships whose validity window
    overlaps the study window are kept, and only those between two listed
    tickers.
    """
    import pandas as pd

    sc = pd.read_parquet("data/revere_supply_chain.parquet")
    permno = pd.read_parquet("data/revere_permno.parquet")

    # company_id -> permno -> ticker. The bridge is CRSP names, not the earnings
    # panel: the earnings panel only covers option-covered firms and resolved
    # 6,758 of 17,600 Revere companies, silently discarding two-thirds of a
    # two-million-edge supply graph. CRSP names resolves 100% of them.
    cid_to_permno = dict(zip(permno["company_id"].astype(str),
                             permno["permno"].astype(int)))
    names = pd.read_parquet("data/crsp_names.parquet",
                            columns=["permno", "crsp_ticker", "nameendt"])
    names = names.dropna(subset=["permno", "crsp_ticker"])
    permno_to_ticker = (names.sort_values("nameendt")
                        .assign(permno=lambda d: d["permno"].astype(int),
                                crsp_ticker=lambda d: d["crsp_ticker"].astype(str).str.upper())
                        .groupby("permno")["crsp_ticker"].last().to_dict())

    def to_ticker(cid):
        p = cid_to_permno.get(str(cid))
        return permno_to_ticker.get(p) if p is not None else None

    lo, hi = window
    sc = sc[(sc["start_"] <= hi) & (sc["end_"] >= lo)]
    edges = []
    seen = set()
    for row in sc.itertuples(index=False):
        s = to_ticker(row.supplier_id)
        d = to_ticker(row.customer_id)
        if not s or not d or s == d:
            continue
        if s not in listed_tickers or d not in listed_tickers:
            continue
        # Timestamp at relationship start, clamped into the study window so an
        # edge that began in 2003 but is still active enters at the window open.
        start = row.start_ if row.start_ >= lo else lo
        ts = int(datetime.strptime(start[:10], "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp())
        key = (s, d)
        if key in seen:
            continue
        seen.add(key)
        rp = row.revenue_percent
        weight = 0.0 if pd.isna(rp) else float(rp) / 100.0
        edges.append((s, d, ts, "supply_chain",
                      _pack(REL_SUPPLY, "supply_chain"), weight))
    if not quiet:
        print(f"  supply_chain: {len(edges):,} listed<->listed edges")
    return edges


def characterise(name, adj):
    if not adj:
        print(f"  {name}: empty")
        return 0
    seen, largest = set(), 0
    for start in adj:
        if start in seen:
            continue
        comp = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp.add(y)
                    stack.append(y)
        seen |= comp
        largest = max(largest, len(comp))
    two_hop = []
    for x in adj:
        reach = set()
        for m in adj[x]:
            reach |= adj[m]
        two_hop.append(len(reach - adj[x] - {x}))
    two_hop.sort()
    deg = sorted(len(v) for v in adj.values())
    print(f"  {name}")
    print(f"     nodes {len(adj):,}  LCC {largest} ({100*largest/len(adj):.0f}%)  "
          f"deg med {deg[len(deg)//2]} p90 {deg[int(.9*len(deg))]}")
    print(f"     two-hop degree: median {two_hop[len(two_hop)//2]}  "
          f"p90 {two_hop[int(.9*len(two_hop))]}  max {two_hop[-1]}")
    return largest


def adjacency(edges):
    adj = defaultdict(set)
    for a, b, *_ in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges-8k", default="data/edgar/edges/targeted/*.jsonl")
    parser.add_argument("--interlock", default="data/form345/2024q[23]_form345.zip")
    args = parser.parse_args()

    from ingestion.edgar import company_tickers
    import pandas as pd
    resolver = EntityResolver(company_tickers())
    # The tradeable universe is every ticker CRSP has returns for, not just the
    # 8k the SEC currently maps -- CRSP covers ~37k permnos, and returns are
    # keyed by permno, so widening here loses nothing downstream and admits far
    # more supply-chain and interlock endpoints.
    crsp_tickers = set(pd.read_parquet("data/crsp_names.parquet",
                                       columns=["crsp_ticker"])["crsp_ticker"]
                       .dropna().astype(str).str.upper())

    print("loading 8-K text edges ...", flush=True)
    e8k = load_8k(args.edges_8k, resolver)
    print(f"  {len(e8k):,} listed<->listed text edges")

    print("loading interlock edges ...", flush=True)
    listed = set(company_tickers()["ticker"]) | crsp_tickers
    einter = load_interlock(args.interlock, listed)
    print(f"  {len(einter):,} listed<->listed interlock edges")

    print("loading supply-chain edges ...", flush=True)
    esupp = load_supply_chain(listed)
    print(f"  {len(esupp):,} listed<->listed supply-chain edges")

    combined = e8k + einter + esupp
    print(f"\n{'='*60}\nSOURCE COMPARISON\n{'='*60}")
    print(f"by source: " + "  ".join(f"{s}={c:,}"
          for s, c in Counter(e[3] for e in combined).most_common()))

    characterise("8-K text only", adjacency(e8k))
    characterise("interlock only", adjacency(einter))
    characterise("supply-chain only", adjacency(esupp))
    characterise("COMBINED", adjacency(combined))

    # Pair-sets per source: the union over the largest single source is the
    # multi-source gain; pairwise overlaps show whether two sources measure the
    # same structure.
    def pairs(edges):
        return {(min(a, b), max(a, b)) for a, b, *_ in edges}
    p8, pi, ps = pairs(e8k), pairs(einter), pairs(esupp)
    union = p8 | pi | ps
    largest = max(len(p8), len(pi), len(ps))
    print(f"\nfirm-pairs by source: text={len(p8):,}  interlock={len(pi):,}  "
          f"supply={len(ps):,}")
    print(f"pairwise overlap: text&inter={len(p8&pi):,}  "
          f"text&supply={len(p8&ps):,}  inter&supply={len(pi&ps):,}")
    print(f"union: {len(union):,}  (+{100*len(union)/largest-100:.0f}% over the "
          f"largest single source)")

    # Do the text-derived and database-derived supplier edges agree? Both claim
    # supplier/customer relationships from independent sources.
    text_supply = {(min(a, b), max(a, b)) for a, b, _, src, code, _ in e8k
                   if (code & 0xFF) == int(Relation.SUPPLIES_TO)}
    if text_supply:
        agree = len(text_supply & ps)
        print(f"\ntext SUPPLIES_TO vs Revere supply-chain:")
        print(f"  {len(text_supply):,} text supplier pairs, {len(ps):,} Revere pairs")
        print(f"  agree on {agree:,} pairs "
              f"({100*agree/len(text_supply):.1f}% of text edges confirmed by Revere)")


if __name__ == "__main__":
    main()
