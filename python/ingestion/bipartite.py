"""
Bipartite temporal graphs: entity <-> bridge <-> entity.

Three studies share this shape. An analyst covers several firms; a dealer trades
with several counterparties; a director sits on several boards. In each case
information plausibly travels from one entity to another *through* the shared
bridge, and the interesting question is whether an event on one side predicts a
move on the other.

The shape is also the reason a temporal graph network is the right tool rather
than a regression on a fixed adjacency matrix:

  the path is irreducibly two hops -- collapsing the bridge into a firm-firm
  edge discards which bridge connected them and when it last touched each side,
  which is the whole mechanism;

  the graph changes continuously as coverage, relationships and board seats
  start and stop, so a point-in-time snapshot is wrong by construction;

  events arrive at arbitrary timestamps rather than on a period grid;

  and new entities and bridges appear constantly, so the model has to be
  inductive rather than fitted per node.

A static method beats this whenever the adjacency is stable and the sampling is
regular -- which is exactly why supply-chain lead-lag is better served by a
spectral decomposition than by anything here.
"""

import numpy as np
import pandas as pd

SCHEMA = ["ts", "entity", "bridge", "relation", "weight"]


def normalise(frame, ts, entity, bridge, relation=None, weight=None):
    """Coerce a source-specific table into the shared schema, sorted by time."""
    out = pd.DataFrame({
        "ts": pd.to_numeric(frame[ts], errors="coerce"),
        "entity": frame[entity].astype(str).str.strip(),
        "bridge": frame[bridge].astype(str).str.strip(),
    })
    out["relation"] = (frame[relation].astype(str).str.strip() if relation else "")
    out["weight"] = (pd.to_numeric(frame[weight], errors="coerce") if weight else 1.0)

    out = out.dropna(subset=["ts", "entity", "bridge"])
    out = out[(out["entity"] != "") & (out["bridge"] != "")]
    out["ts"] = out["ts"].astype("int64")
    return out.sort_values("ts", kind="stable").reset_index(drop=True)[SCHEMA]


def density_report(edges, window="QE", min_shared=1, quiet=False):
    """
    The go/no-go check, run before any modelling.

    Two graphs in this project died here: a supplier network at median degree
    1.7 and an earnings-call mention graph at 1.9. Attention over a
    one-element neighbourhood is a linear map, so a graph that cannot clear
    roughly five neighbours per period is not worth a temporal GNN whatever
    else is true about it.

    Reports both hops. `bridge_degree` is how many entities a bridge touches --
    a bridge touching one entity connects nothing. `two_hop_degree` is how many
    other entities an entity reaches through shared bridges, which is the
    number that actually decides feasibility.
    """
    frame = edges.copy()
    frame["period"] = pd.to_datetime(frame["ts"], unit="s").dt.to_period(
        "Q" if window in ("QE", "Q") else window)

    bridge_degree = frame.groupby(["period", "bridge"])["entity"].nunique()
    entity_degree = frame.groupby(["period", "entity"])["bridge"].nunique()

    # Two-hop reach: entities sharing at least one bridge inside a period.
    reach = []
    for period, chunk in frame.groupby("period"):
        pairs = chunk[["entity", "bridge"]].drop_duplicates()
        counts = pairs.groupby("bridge")["entity"].nunique()
        live = set(counts[counts > 1].index)
        connected = pairs[pairs["bridge"].isin(live)]
        per_entity = (connected.merge(connected, on="bridge")
                      .query("entity_x != entity_y")
                      .groupby("entity_x")["entity_y"].nunique())
        reach.append(per_entity)
    two_hop = pd.concat(reach) if reach else pd.Series(dtype=float)

    stats = {
        "edges": len(frame),
        "entities": frame["entity"].nunique(),
        "bridges": frame["bridge"].nunique(),
        "periods": frame["period"].nunique(),
        "bridge_degree_median": float(bridge_degree.median()),
        "bridge_degree_mean": float(bridge_degree.mean()),
        "multi_entity_bridge_share": float((bridge_degree > 1).mean()),
        "entity_degree_median": float(entity_degree.median()),
        "two_hop_degree_median": float(two_hop.median()) if len(two_hop) else 0.0,
        "two_hop_degree_mean": float(two_hop.mean()) if len(two_hop) else 0.0,
        "two_hop_ge5_share": float((two_hop >= 5).mean()) if len(two_hop) else 0.0,
    }

    if not quiet:
        print(f"  edges {stats['edges']:,} | entities {stats['entities']:,} | "
              f"bridges {stats['bridges']:,} | periods {stats['periods']}")
        print(f"  bridge degree per period : median {stats['bridge_degree_median']:.0f} "
              f"mean {stats['bridge_degree_mean']:.1f} | "
              f"{stats['multi_entity_bridge_share']:.0%} of bridges touch >1 entity")
        print(f"  TWO-HOP degree per period: median {stats['two_hop_degree_median']:.0f} "
              f"mean {stats['two_hop_degree_mean']:.1f} | "
              f"{stats['two_hop_ge5_share']:.0%} reach >=5 peers")
        verdict = ("VIABLE" if stats["two_hop_degree_median"] >= 5
                   else "MARGINAL" if stats["two_hop_degree_median"] >= 3 else "TOO SPARSE")
        print(f"  -> {verdict} for temporal attention")
    return stats
