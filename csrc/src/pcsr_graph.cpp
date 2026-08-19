#include "pcsr_graph.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

// Layout recap: the out-edges of vertex v live in edges[vertex_offsets[v] ..
// vertex_offsets[v+1]). The first vertex_counts[v] slots of that region are
// live edges; everything after is a gap holding EMPTY_GAP.

namespace {

constexpr uint32_t NO_HOT_VERTEX = 0xFFFFFFFF;

/**
 * Spread `spare` free slots over `n` vertices, weighted by (count + 1).
 *
 * Two things are going on here.
 *
 * First, `hot` -- the vertex whose overflow triggered this rebalance -- gets
 * first claim on enough slots to double its run. Without that, a vertex is
 * handed only its proportional share, which on a power-law graph is a rounding
 * error next to the (count + 1) mass of the tens of thousands of degree-0
 * vertices sharing the window. It then overflows again a few inserts later, and
 * building one high-degree vertex costs O(degree) rebalances instead of
 * O(log degree). Doubling makes each rebalance buy geometrically more room, so
 * the rewrites telescope into an amortized constant.
 *
 * Second, the remainder is weighted by degree so busy vertices keep more
 * headroom than idle ones. The +1 guarantees currently-empty vertices still
 * receive slots -- otherwise their first edge would immediately rebalance.
 *
 * Distribution is exact (sum(gaps) == spare), so no slot is ever stranded.
 */
void distribute_gaps(const uint32_t* counts, uint32_t n, uint64_t total,
                     uint32_t spare, uint32_t* gaps, uint32_t hot) {
    uint32_t reserved = 0;
    if (hot < n) {
        reserved = static_cast<uint32_t>(std::min<uint64_t>(spare, counts[hot]));
        spare -= reserved;
    }

    const uint64_t weight_total = total + n; // sum of (count_i + 1)
    uint64_t assigned = 0;

    for (uint32_t i = 0; i < n; ++i) {
        gaps[i] = static_cast<uint32_t>(
            (static_cast<uint64_t>(spare) * (counts[i] + 1)) / weight_total);
        assigned += gaps[i];
    }

    // Each floor division discards < 1 slot, so the shortfall is < n.
    uint32_t leftover = static_cast<uint32_t>(spare - assigned);
    for (uint32_t i = 0; i < leftover; ++i) {
        ++gaps[i];
    }

    if (hot < n) {
        gaps[hot] += reserved;
    }
}

void fill_gaps(TemporalEdge* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = {EMPTY_GAP, 0};
    }
}

} // namespace

PCSRGraph::PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity, size_t arena_bytes)
    : num_vertices(max_vertices),
      edge_capacity(initial_edge_capacity < max_vertices ? max_vertices : initial_edge_capacity),
      num_edges(0),
      rebalance_count(0),
      resize_count(0),
      slots_rewritten(0),
      arena(arena_bytes) {

    if (num_vertices == 0) {
        throw std::invalid_argument("PCSRGraph: max_vertices must be non-zero");
    }

    // The arena already rounds every block up to a 64-byte boundary.
    vertex_offsets = static_cast<uint32_t*>(
        arena.allocate((static_cast<size_t>(num_vertices) + 1) * sizeof(uint32_t)));
    vertex_counts = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));
    edges = static_cast<TemporalEdge*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(TemporalEdge)));
    scratchpad_edges = static_cast<TemporalEdge*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(TemporalEdge)));
    scratchpad_counts = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));
    scratchpad_gaps = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));

    fill_gaps(edges, edge_capacity);
    std::memset(vertex_counts, 0, static_cast<size_t>(num_vertices) * sizeof(uint32_t));

    // Empty graph: every slot is spare, so this hands each vertex an equal
    // region (plus one extra to the first `edge_capacity % num_vertices`).
    distribute_gaps(vertex_counts, num_vertices, 0, edge_capacity, scratchpad_gaps,
                    NO_HOT_VERTEX);

    uint32_t cursor = 0;
    for (uint32_t v = 0; v < num_vertices; ++v) {
        vertex_offsets[v] = cursor;
        cursor += scratchpad_gaps[v];
    }
    vertex_offsets[num_vertices] = edge_capacity;
}

PCSRGraph::~PCSRGraph() {
}

void PCSRGraph::insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp) {
    if (src >= num_vertices || dst >= num_vertices) {
        throw std::out_of_range("requested node is nonexistent. nodes go from 0 to " +
                                std::to_string(num_vertices - 1) + ".");
    }

    const uint32_t start = vertex_offsets[src];
    const uint32_t region = vertex_offsets[src + 1] - start;
    const uint32_t count = vertex_counts[src];

    // Fast path: regions stay left-packed, so the next free slot is known
    // without scanning. This is the O(1) common case.
    if (count < region) [[likely]] {
        edges[start + count] = {dst, timestamp};
        vertex_counts[src] = count + 1;
        ++num_edges;
        return;
    }

    rebalance_and_insert(src, dst, timestamp);
}

void PCSRGraph::rebalance_and_insert(uint32_t src, uint32_t dst, uint32_t timestamp) {
    // Walk up power-of-two windows of vertices centred on src's aligned block
    // until one is loose enough to absorb the insert. Doubling (rather than
    // widening by one vertex at a time) is what makes the amortized cost
    // logarithmic instead of linear in V.
    uint32_t win_nodes = 1;

    while (true) {
        win_nodes <<= 1;
        const bool full_array = (win_nodes >= num_vertices);

        uint32_t v_start, v_end;
        if (full_array) {
            v_start = 0;
            v_end = num_vertices - 1;
        } else {
            v_start = (src / win_nodes) * win_nodes;
            v_end = v_start + win_nodes - 1;
            if (v_end >= num_vertices) {
                v_end = num_vertices - 1;
            }
        }

        const uint32_t win_st = vertex_offsets[v_start];
        const uint32_t win_end = vertex_offsets[v_end + 1];
        const uint64_t win_capacity = win_end - win_st;

        uint64_t occupied = 1; // the edge we are about to add
        for (uint32_t v = v_start; v <= v_end; ++v) {
            occupied += vertex_counts[v];
        }

        // Upper density bound. Leaving >= 25% of the window free means the
        // vertices in it can absorb many more inserts before the next
        // rebalance, which is what amortizes the rewrite cost.
        //
        // The same bound has to apply at the top level, and that is load
        // bearing: if a full array were allowed to rebalance at 100% density it
        // would hand out almost no gaps, refill immediately, and rewrite all of
        // memory again on the next insert -- quadratic. Exceeding the bound
        // with nowhere left to escalate is precisely the signal to grow.
        const uint64_t limit = (win_capacity * 3) / 4;

        if (occupied <= limit) {
            redistribute(v_start, v_end, src, dst, timestamp);
            return;
        }

        if (full_array) [[unlikely]] {
            resize_pma(src);
            insert_edge(src, dst, timestamp); // retry against the doubled PMA
            return;
        }
    }
}

void PCSRGraph::redistribute(uint32_t v_start, uint32_t v_end,
                             uint32_t src, uint32_t dst, uint32_t timestamp) {
    const uint32_t win_st = vertex_offsets[v_start];
    const uint32_t win_end = vertex_offsets[v_end + 1];
    const uint32_t win_capacity = win_end - win_st;
    const uint32_t n = v_end - v_start + 1;

    ++rebalance_count;
    slots_rewritten += win_capacity;

    // Pass 1: lift every live edge in the window into the scratchpad, in
    // vertex order, splicing the new edge into src's run as we pass it.
    uint32_t scratch_idx = 0;
    for (uint32_t v = v_start; v <= v_end; ++v) {
        const uint32_t region_start = vertex_offsets[v];
        uint32_t count = vertex_counts[v];

        for (uint32_t i = 0; i < count; ++i) {
            scratchpad_edges[scratch_idx++] = edges[region_start + i];
        }
        if (v == src) {
            scratchpad_edges[scratch_idx++] = {dst, timestamp};
            ++count;
        }
        scratchpad_counts[v - v_start] = count;
    }

    const uint32_t total = scratch_idx;

    // Guaranteed by the density check in rebalance_and_insert, but this is the
    // invariant whose violation silently ate edges in the previous version.
    if (total > win_capacity) {
        throw std::logic_error("PCSRGraph: rebalance window overflow");
    }

    // Pass 2: lay the runs back down, each followed by its share of the gaps.
    fill_gaps(edges + win_st, win_capacity);
    distribute_gaps(scratchpad_counts, n, total, win_capacity - total, scratchpad_gaps,
                    src - v_start);

    uint32_t cursor = win_st;
    uint32_t consumed = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t count = scratchpad_counts[i];

        vertex_offsets[v_start + i] = cursor;
        vertex_counts[v_start + i] = count;
        for (uint32_t j = 0; j < count; ++j) {
            edges[cursor + j] = scratchpad_edges[consumed++];
        }
        cursor += count + scratchpad_gaps[i];
    }

    // sum(counts) + sum(gaps) == total + (win_capacity - total) == win_capacity,
    // so the window closes exactly where it started and vertices outside it are
    // untouched.
    vertex_offsets[v_end + 1] = win_end;
    ++num_edges;
}

void PCSRGraph::resize_pma(uint32_t hot) {
    const uint64_t wanted = static_cast<uint64_t>(edge_capacity) * 2;
    if (wanted > 0xFFFFFFFFull) {
        throw std::runtime_error("PCSRGraph: edge capacity would exceed 2^32 slots");
    }
    const uint32_t new_capacity = static_cast<uint32_t>(wanted);

    ++resize_count;
    slots_rewritten += new_capacity;

    // Allocated from the arena, not aligned_alloc. The old call site leaked on
    // every growth and, on macOS, returned NULL whenever the byte count was not
    // a multiple of 64 (aligned_alloc requires that) and then wrote through it.
    TemporalEdge* new_edges = static_cast<TemporalEdge*>(
        arena.allocate(static_cast<size_t>(new_capacity) * sizeof(TemporalEdge)));
    TemporalEdge* new_scratch = static_cast<TemporalEdge*>(
        arena.allocate(static_cast<size_t>(new_capacity) * sizeof(TemporalEdge)));
    // A fresh offsets array: the old one has to stay readable while we walk it.
    uint32_t* new_offsets = static_cast<uint32_t*>(
        arena.allocate((static_cast<size_t>(num_vertices) + 1) * sizeof(uint32_t)));

    fill_gaps(new_edges, new_capacity);

    const uint32_t spare = new_capacity - static_cast<uint32_t>(num_edges);
    distribute_gaps(vertex_counts, num_vertices, num_edges, spare, scratchpad_gaps, hot);

    uint32_t cursor = 0;
    for (uint32_t v = 0; v < num_vertices; ++v) {
        const uint32_t old_start = vertex_offsets[v];
        const uint32_t count = vertex_counts[v];

        new_offsets[v] = cursor;
        for (uint32_t i = 0; i < count; ++i) {
            new_edges[cursor + i] = edges[old_start + i];
        }
        cursor += count + scratchpad_gaps[v];
    }
    new_offsets[num_vertices] = new_capacity;

    edges = new_edges;
    scratchpad_edges = new_scratch;
    vertex_offsets = new_offsets;
    edge_capacity = new_capacity;
}
