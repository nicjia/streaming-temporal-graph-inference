"""
Streams a historical GDELT event table into the C++ PCSR engine.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import graph_engine  # noqa: E402

from ingestion.id_mapper import EntityMapper  # noqa: E402


def provision(num_edges, num_nodes, slack=2.0, slots_per_vertex=16):
    """
    Choose a PMA capacity.

    Two floors, both of which matter. `slack` covers total edges: a PMA with no
    free slots has nowhere to insert. `slots_per_vertex` covers the per-vertex
    regions, which is the one people forget -- with more vertices than the slack
    alone accounts for, each region gets a handful of slots and even modest
    degrees rebalance constantly. Under-provisioning no longer loses edges, but
    it does multiply the work per insert by two to three orders of magnitude
    (see benchmarks/cpp_perf_test.cpp for the slots-rewritten-per-edge figures).
    """
    return max(int(num_edges * slack), int(num_nodes * slots_per_vertex))


class DataReplayer:
    """
    Wraps a PCSRGraph and feeds event tables into it.

    Args:
        num_nodes: Vertex capacity. Also the natural cap for the entity mapper.
        max_edges: Initial PMA slot count.
        arena_bytes: Arena budget. Growth re-allocates without reclaiming the
            old blocks, so leave headroom if you expect the PMA to grow.
    """

    def __init__(self, num_nodes, max_edges, arena_bytes=128 * 1024 * 1024):
        print(f"Initializing C++ Engine: {num_nodes:,} nodes, {max_edges:,} capacity...")
        self.num_nodes = num_nodes
        self.graph = graph_engine.PCSRGraph(num_nodes, max_edges, int(arena_bytes))

    @classmethod
    def for_events(cls, num_events, num_nodes, slack=2.0, slots_per_vertex=16,
                   arena_multiple=4):
        """Build a replayer sized for a known event count."""
        capacity = provision(num_events, num_nodes, slack, slots_per_vertex)
        arena = max(256 * 1024 * 1024, capacity * 8 * arena_multiple)
        return cls(num_nodes, capacity, arena)

    def replay_events(self, src_names, dst_names, timestamps, mapper=None,
                      quiet=False):
        """
        Map entity strings to ids and stream the events into the engine.

        Args:
            src_names, dst_names: Sequences of entity strings (or integer ids
                already, if mapper is None).
            timestamps: Unix seconds, int-like.
            mapper: EntityMapper. If None, the name arrays are taken as ids.

        Returns:
            dict of replay statistics.
        """
        if mapper is None:
            src_ids = np.asarray(src_names, dtype=np.int64)
            dst_ids = np.asarray(dst_names, dtype=np.int64)
        else:
            # Bulk id mapping: resolves each distinct string once rather than
            # once per row. Feeding a bulk insert from a row-at-a-time mapper
            # would just move the bottleneck.
            src_ids = mapper.get_ids(src_names)
            dst_ids = mapper.get_ids(dst_names)

        timestamps = np.asarray(timestamps, dtype=np.int64)

        # -1 marks a blank name or an entity past the mapper's cap. Ids at or
        # beyond the vertex count would raise out_of_range inside the engine.
        keep = ((src_ids >= 0) & (dst_ids >= 0) &
                (src_ids < self.num_nodes) & (dst_ids < self.num_nodes))
        dropped = int((~keep).sum())

        src_ids = src_ids[keep]
        dst_ids = dst_ids[keep]
        timestamps = timestamps[keep]

        # Chronological order is not cosmetic: the temporal sampler binary-
        # searches each adjacency run, which is only valid if runs are sorted,
        # and runs are sorted only if events go in sorted. Stable sort keeps
        # same-timestamp events in their original order.
        order = np.argsort(timestamps, kind="stable")
        src_ids = src_ids[order].astype(np.uint32)
        dst_ids = dst_ids[order].astype(np.uint32)
        timestamps = timestamps[order].astype(np.uint32)

        start = time.perf_counter()
        self.graph.insert_edges(src_ids, dst_ids, timestamps)
        elapsed = time.perf_counter() - start

        stats = {
            "inserted": int(len(src_ids)),
            "dropped": dropped,
            "seconds": elapsed,
            "edges_per_second": (len(src_ids) / elapsed) if elapsed > 0 else float("inf"),
            "ns_per_edge": (elapsed * 1e9 / len(src_ids)) if len(src_ids) else 0.0,
            "graph_edges": int(self.graph.num_edges),
            "rebalances": int(self.graph.rebalance_count),
            "resizes": int(self.graph.resize_count),
            "slots_rewritten_per_edge": (self.graph.slots_rewritten / len(src_ids)
                                         if len(src_ids) else 0.0),
        }

        if not quiet:
            print(f"Streamed {stats['inserted']:,} events in {elapsed * 1e3:.2f} ms "
                  f"({stats['edges_per_second']:,.0f} events/s, "
                  f"{stats['ns_per_edge']:.1f} ns/event)")
            if dropped:
                print(f"  dropped {dropped:,} events with unmapped or out-of-range actors")
            print(f"  graph now holds {stats['graph_edges']:,} edges "
                  f"({stats['rebalances']:,} rebalances, {stats['resizes']} resizes, "
                  f"{stats['slots_rewritten_per_edge']:.2f} slots rewritten/edge)")

        return stats

    def replay_dataframe(self, df, mapper, src_col="Actor1Name", dst_col="Actor2Name",
                         time_col="DATEADDED", quiet=False):
        """Replay an in-memory event table."""
        df = df.dropna(subset=[src_col, dst_col, time_col])
        timestamps = to_unix_seconds(df[time_col])
        return self.replay_events(df[src_col].to_numpy(), df[dst_col].to_numpy(),
                                  timestamps, mapper, quiet=quiet)

    def replay_gdelt_csv(self, csv_path, mapper, quiet=False):
        """Replay a GDELT CSV written by DataFetcher."""
        if not quiet:
            print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)
        self.replay_dataframe(df, mapper, quiet=quiet)
        return self.graph


def to_unix_seconds(column):
    """
    Convert GDELT's DATEADDED (YYYYMMDDHHMMSS) to Unix seconds.

    Already-Unix values pass through: a corpus that has been through this once
    should not be reinterpreted as a calendar literal on the second pass.
    """
    values = pd.to_numeric(column, errors="coerce")

    # 1e13 sits above any plausible Unix timestamp in seconds and below the
    # smallest YYYYMMDDHHMMSS literal (1e13 == year 1000).
    if values.dropna().empty or values.max() < 1e13:
        return values.astype("int64").to_numpy()

    parsed = pd.to_datetime(values.astype("int64").astype(str),
                            format="%Y%m%d%H%M%S", errors="coerce")

    # Subtract the epoch and divide by a Timedelta rather than casting to
    # int64 and dividing by 10**9. pandas 2.x resolves this parse to
    # datetime64[us], not [ns], so the fixed divisor silently returned
    # timestamps a thousand times too small -- every event landed in 1970.
    return ((parsed - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy()
