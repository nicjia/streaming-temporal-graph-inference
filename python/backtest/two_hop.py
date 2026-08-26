"""
Two-hop causal neighbourhoods over a bipartite temporal graph.

Given entities connected through bridges (firms through directors, firms through
analysts, counterparties through dealers), answer: which other entities did this
entity share a bridge with, as of a given instant?

The "as of" is the whole point, and it is why this runs on the graph engine
rather than on a dictionary. Board seats, coverage and trading relationships
start and stop; asking who shared a director with Apple *today* when studying a
2019 event is a lookahead bug that no amount of careful return alignment
repairs. The engine's sampler only reads edges stamped strictly before the query
time, so the neighbourhood is correct by construction at every timestamp.

Entities and bridges share one integer id space so a single two-hop sample walks
entity -> bridge -> entity in one pass.
"""

import numpy as np
import pandas as pd

import graph_engine
from models import PCSRTemporalSampler


class BipartiteNeighborhood:
    """
    Causal two-hop lookup over an entity <-> bridge graph.

    Args:
        edges: DataFrame with [ts, entity, bridge] (the bipartite SCHEMA).
        fanout: neighbours sampled per hop. The product bounds how many peers
            a single query can return.
    """

    def __init__(self, edges, fanout=32, arena_multiple=6, quiet=False):
        entities = pd.Index(sorted(edges["entity"].unique()))
        bridges = pd.Index(sorted(edges["bridge"].unique()))

        # One id space, entities first, so a returned id under len(entities) is
        # an entity and anything above it is a bridge.
        self.entities = entities
        self.n_entities = len(entities)
        self.entity_id = {name: i for i, name in enumerate(entities)}
        self.bridge_id = {name: self.n_entities + i for i, name in enumerate(bridges)}
        total = self.n_entities + len(bridges)

        src = edges["entity"].map(self.entity_id).to_numpy(np.int64)
        dst = edges["bridge"].map(self.bridge_id).to_numpy(np.int64)
        ts = edges["ts"].to_numpy(np.int64)

        # Undirected: store both directions so one hop traverses either way.
        both_src = np.concatenate([src, dst]).astype(np.uint32)
        both_dst = np.concatenate([dst, src]).astype(np.uint32)
        both_ts = np.concatenate([ts, ts])
        order = np.argsort(both_ts, kind="stable")

        capacity = max(int(len(both_src) * 2.5), total * 8)
        self.graph = graph_engine.PCSRGraph(total, capacity,
                                            capacity * 8 * arena_multiple + (1 << 26))
        self.graph.insert_edges(both_src[order], both_dst[order],
                                both_ts[order].astype(np.uint32), None)
        self.sampler = PCSRTemporalSampler(self.graph)
        self.fanout = fanout

        if not quiet:
            print(f"Bipartite graph: {self.n_entities:,} entities + {len(bridges):,} bridges, "
                  f"{self.graph.num_edges:,} directed edges, "
                  f"chronological: {self.sampler.validate()}")

    def peers(self, entity, timestamp):
        """Entities sharing a bridge with `entity` strictly before `timestamp`."""
        node = self.entity_id.get(entity)
        if node is None:
            return []
        stamp = int(timestamp)

        ids, _, mask = self.sampler.sample(np.array([node]), np.array([stamp]),
                                           self.fanout)
        hop1 = ids[0][mask[0]]
        hop1 = hop1[hop1 >= self.n_entities]        # bridges only
        if len(hop1) == 0:
            return []

        ids2, _, mask2 = self.sampler.sample(hop1.astype(np.int64),
                                             np.full(len(hop1), stamp), self.fanout)
        hop2 = np.unique(ids2[mask2])
        hop2 = hop2[hop2 < self.n_entities]         # entities only
        hop2 = hop2[hop2 != node]                   # drop the source
        return [self.entities[i] for i in hop2]

    def peer_map(self, entities, timestamps):
        """Vectorised over a batch; returns a list of peer lists."""
        return [self.peers(e, t) for e, t in zip(entities, timestamps)]
