"""
Temporal neighbourhood sampling straight out of the PCSR arena.

This is where the C++ engine pays for itself. The sampler never copies the
graph: it holds the zero-copy numpy views over the PMA and does all of its work
with vectorised index arithmetic on them.
"""

import numpy as np

EMPTY_GAP = 0xFFFFFFFF


class PCSRTemporalSampler:
    """
    Answers "which neighbours had this node interacted with strictly before
    time t?" for a whole batch at once.

    Two properties of the PCSR make this cheap.

    First, each vertex's adjacency is a single contiguous, gap-free run --
    regions are left-packed, so vertex v's live edges are exactly
    edges[offsets[v] : offsets[v] + counts[v]]. No gap-skipping, and the run is
    usually one or two cache lines.

    Second, if events were inserted in chronological order then every run is
    sorted ascending by timestamp, because rebalancing and growth both preserve
    insertion order within a vertex (asserted in tests/test_pcsr.cpp). That
    turns "events before t" into a binary search instead of a scan.

    The binary search is done across the whole batch at once. numpy's
    searchsorted needs one sorted array, but each vertex's run is a separate
    sorted array at a different offset, so the bisection is written out
    explicitly -- ceil(log2(max_degree)) vectorised steps over the batch.

    Args:
        graph: A graph_engine.PCSRGraph.
        assume_sorted: Set False if events were inserted out of order; the
            sampler then falls back to a scan-and-sort per query, which is
            correct but much slower. Use validate() to check.
    """

    def __init__(self, graph, assume_sorted=True):
        self.graph = graph
        self.assume_sorted = assume_sorted
        self.refresh()

    def refresh(self):
        """
        Re-acquire the views.

        Required after any insert that grew the PMA: resize_pma() reseats the
        internal pointers, and a numpy array handed out earlier still aliases
        the old, now-abandoned arena block. It would read stale data rather
        than fail, so re-acquiring is not optional.
        """
        self.offsets = np.asarray(self.graph.get_vertex_offsets())
        self.counts = np.asarray(self.graph.get_vertex_counts())
        edges = np.asarray(self.graph.get_edges())
        self.edge_targets = edges[:, 0]
        self.edge_times = edges[:, 1]
        self.edge_relations = np.asarray(self.graph.get_edge_relations())
        self.num_vertices = int(self.graph.num_vertices)

    def validate(self):
        """True if every adjacency run is ascending in time."""
        for v in range(self.num_vertices):
            start = self.offsets[v]
            times = self.edge_times[start:start + self.counts[v]]
            if times.size > 1 and np.any(np.diff(times.astype(np.int64)) < 0):
                return False
        return True

    def _cut_indices(self, nodes, times):
        """
        For each (node, time) query, the index of the first edge in that node's
        run with timestamp >= time. Everything from the run's start up to that
        index is causally admissible.

        Strictly-before, not before-or-equal, and that matters here: GDELT
        stamps every event in a 15-minute batch with the same DATEADDED. If the
        cut were inclusive, predicting an edge at time t would get to look at
        that very edge -- and at every other edge in its bucket -- as evidence.
        The model would score beautifully and have learned nothing.
        """
        starts = self.offsets[nodes]
        ends = starts + self.counts[nodes]

        lo = starts.copy()
        hi = ends.copy()

        max_degree = int(self.counts.max()) if self.counts.size else 0
        steps = max(1, int(np.ceil(np.log2(max_degree + 1))) + 1)

        for _ in range(steps):
            active = lo < hi
            if not active.any():
                break
            mid = (lo + hi) >> 1
            # Reads at inactive lanes are harmless (mid stays in bounds) and
            # branchless beats masking the gather.
            before = self.edge_times[np.minimum(mid, self.edge_times.size - 1)] < times
            lo = np.where(active & before, mid + 1, lo)
            hi = np.where(active & ~before, mid, hi)

        return starts, lo

    def recency_features(self, nodes, times, num_neighbors):
        """
        Scalar rate features for each (node, time) query.

        These exist because the attention layer cannot produce them. Attention
        pools its neighbourhood with softmax weights, which is a weighted
        *average*; a mean aggregator cannot distinguish two histories that
        differ only in how many events they contain, and the LayerNorm on the
        layer output removes what little magnitude survives. So a TGAT
        embedding can tell you who a country interacts with and cannot tell you
        how often -- measured at 99.6% static variance when asked for a rate.

        The information is right there in the PMA: the cut index gives the size
        of the admissible history, and the timestamps around it give the recent
        arrival rate. Computing it here and concatenating it to the embedding
        costs one extra bisection and lets the head see cardinality.

        All four are expressed in *day* units and scaled to be O(1). That is
        not cosmetic. The first version returned log-seconds, so the head was
        fed values around 8 to 12 with small variation on top; a two-layer MLP
        given inputs like that trains badly, and it converged to a fit whose
        output correlated -0.51 with the target it was trained on -- inverted,
        while still matching the mean. The features themselves were correct
        throughout (log rate alone correlates +0.58 with future counts). Scale
        is the whole difference.

        Returns:
            (B, 4) float32: scaled history size, log1p(days since the most
            recent event), log1p(days spanned by the last K events), and the
            log arrival rate in events per day over that span.
        """
        seconds_per_day = 86400.0
        nodes = np.asarray(nodes, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)

        starts, cuts = self._cut_indices(nodes, times)
        available = cuts - starts
        has_history = available > 0

        # Index of the most recent admissible event, and of the K-th most
        # recent, both clamped into the node's own region.
        last_index = np.clip(cuts - 1, starts, None)
        kth_index = np.maximum(starts, cuts - num_neighbors)

        last_time = self.edge_times[np.clip(last_index, 0, self.edge_times.size - 1)]
        kth_time = self.edge_times[np.clip(kth_index, 0, self.edge_times.size - 1)]

        # No history: report a maximally stale, zero-rate state rather than a
        # negative or undefined one.
        stale = np.float64(10 ** 7)  # ~116 days
        since_last = np.where(has_history, times - last_time, stale).astype(np.float64)
        span = np.where(has_history, times - kth_time, stale).astype(np.float64)

        since_last_days = np.maximum(since_last, 0.0) / seconds_per_day
        span_days = np.maximum(span / seconds_per_day, 1.0 / 24.0)

        counted = np.minimum(available, num_neighbors).astype(np.float64)
        rate_per_day = np.log((counted + 1.0) / span_days)

        features = np.stack([
            np.log1p(available.astype(np.float64)) / 5.0,
            np.log1p(since_last_days),
            np.log1p(span_days),
            rate_per_day,
        ], axis=1)
        return features.astype(np.float32)

    def sample(self, nodes, times, num_neighbors, strategy="recent", rng=None,
               with_relations=False):
        """
        Args:
            nodes: (B,) vertex ids.
            times: (B,) query timestamps.
            num_neighbors: K, the fixed fan-out per node.
            strategy: "recent" takes the K most recent admissible edges;
                "uniform" samples K of them at random with replacement.
            rng: numpy Generator, required for "uniform".

        Returns:
            neighbor_ids: (B, K) uint32, zero where masked.
            neighbor_times: (B, K) int64, zero where masked.
            mask: (B, K) bool, True where the slot holds a real neighbour.
            neighbor_relations: (B, K) uint16, zero where masked. Only returned
                when `with_relations` is set, so existing two-hop callers keep
                their three-tuple.

        Nodes with no admissible history come back fully masked rather than
        dropped, so the batch keeps a fixed shape and the attention layer can
        handle "no history" as its own case.
        """
        nodes = np.asarray(nodes, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        if nodes.shape != times.shape:
            raise ValueError("nodes and times must have the same shape")

        starts, cuts = self._cut_indices(nodes, times)
        available = cuts - starts

        if strategy == "recent":
            # Walk back K slots from the cut. Because runs are time-ordered,
            # the K slots immediately before the cut are the K most recent
            # admissible events -- no sort, no scan.
            offsets_back = np.arange(num_neighbors, dtype=np.int64) - num_neighbors
            idx = cuts[:, None] + offsets_back[None, :]
            mask = idx >= starts[:, None]
        elif strategy == "uniform":
            if rng is None:
                raise ValueError('strategy="uniform" needs an rng')
            # Sampling the *whole* history rather than the tail matters for
            # high-degree hubs, whose recent window is a single news cycle.
            draws = rng.random((nodes.size, num_neighbors))
            idx = starts[:, None] + (draws * np.maximum(available, 1)[:, None]).astype(np.int64)
            # Broadcast to (B, K): every slot of a node with history is valid,
            # every slot of a node without it is not.
            mask = np.repeat(available[:, None] > 0, num_neighbors, axis=1)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        # Clamp before gathering so masked lanes read a valid address instead
        # of reaching into another vertex's region or off the end of the PMA.
        safe = np.clip(idx, 0, self.edge_targets.size - 1)

        neighbor_ids = np.where(mask, self.edge_targets[safe], 0).astype(np.uint32)
        neighbor_times = np.where(mask, self.edge_times[safe], 0).astype(np.int64)

        if not with_relations:
            return neighbor_ids, neighbor_times, mask

        neighbor_relations = np.where(mask, self.edge_relations[safe], 0).astype(np.uint16)
        return neighbor_ids, neighbor_times, mask, neighbor_relations
