#include "pcsr_graph.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

// Layout recap: the out-edges of vertex v live in edges[vertex_offsets[v] ..
// vertex_offsets[v+1]). Within that region the first vertex_starts[v] slots
// hold expired edges, the next vertex_counts[v] hold live ones, and everything
// after is a gap holding EMPTY_GAP. vertex_starts is zero everywhere until
// expire_before() is called, so a graph that never expires has exactly the
// layout it had before expiry existed.

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

PCSRGraph::PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity,
                     size_t arena_bytes, bool store_weights)
    : num_vertices(max_vertices),
      edge_capacity(initial_edge_capacity < max_vertices ? max_vertices : initial_edge_capacity),
      num_edges(0),
      rebalance_count(0),
      resize_count(0),
      slots_rewritten(0),
      expired_edges(0),
      dead_slots(0),
      writer_active(false),
      arena(arena_bytes) {

    if (num_vertices == 0) {
        throw std::invalid_argument("PCSRGraph: max_vertices must be non-zero");
    }

    // The arena already rounds every block up to a 64-byte boundary.
    vertex_offsets = static_cast<uint32_t*>(
        arena.allocate((static_cast<size_t>(num_vertices) + 1) * sizeof(uint32_t)));
    vertex_counts = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));
    vertex_starts = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));
    edges = static_cast<TemporalEdge*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(TemporalEdge)));
    edge_relations = static_cast<EdgeRelation*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(EdgeRelation)));
    edge_weights = store_weights ? static_cast<float*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(float))) : nullptr;
    scratchpad_edges = static_cast<TemporalEdge*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(TemporalEdge)));
    scratchpad_relations = static_cast<EdgeRelation*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(EdgeRelation)));
    scratchpad_weights = store_weights ? static_cast<float*>(
        arena.allocate(static_cast<size_t>(edge_capacity) * sizeof(float))) : nullptr;
    scratchpad_counts = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));
    scratchpad_gaps = static_cast<uint32_t*>(
        arena.allocate(static_cast<size_t>(num_vertices) * sizeof(uint32_t)));

    fill_gaps(edges, edge_capacity);
    // memset writes bytes, so this is only a correct fill because
    // RELATION_UNKNOWN is zero. Asserted rather than assumed.
    static_assert(RELATION_UNKNOWN == 0, "byte-fill assumes a zero sentinel");
    std::memset(edge_relations, 0,
                static_cast<size_t>(edge_capacity) * sizeof(EdgeRelation));
    if (edge_weights) {
        // 0.0f is all-zero bytes under IEEE-754, which every target here uses.
        std::memset(edge_weights, 0, static_cast<size_t>(edge_capacity) * sizeof(float));
    }
    std::memset(vertex_counts, 0, static_cast<size_t>(num_vertices) * sizeof(uint32_t));
    std::memset(vertex_starts, 0, static_cast<size_t>(num_vertices) * sizeof(uint32_t));

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

void PCSRGraph::insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp,
                            EdgeRelation relation, float weight) {
    if (src >= num_vertices || dst >= num_vertices) {
        throw std::out_of_range("requested node is nonexistent. nodes go from 0 to " +
                                std::to_string(num_vertices - 1) + ".");
    }

    const uint32_t start = vertex_offsets[src];
    const uint32_t region = vertex_offsets[src + 1] - start;
    const uint32_t count = vertex_counts[src];
    // Expired edges still hold their slots, so the write position is past them.
    // used == count on any graph that has never expired.
    const uint32_t used = vertex_starts[src] + count;

    // Fast path: regions stay left-packed, so the next free slot is known
    // without scanning. This is the O(1) common case.
    // Rejecting rather than dropping. The compare is against a register and
    // stays off the memory path, so it costs nothing measurable; silently
    // discarding a weight would not show up until someone read zeros back.
    if (weight != 0.0f && !edge_weights) [[unlikely]] {
        throw std::invalid_argument(
            "PCSRGraph: this graph was built without weights, so a non-zero "
            "weight cannot be stored. Construct it with store_weights=true.");
    }

    if (used < region) [[likely]] {
        edges[start + used] = {dst, timestamp};
        edge_relations[start + used] = relation;
        if (edge_weights) edge_weights[start + used] = weight;
        vertex_counts[src] = count + 1;
        ++num_edges;
        return;
    }

    rebalance_and_insert(src, dst, timestamp, relation, weight);
}

void PCSRGraph::rebalance_and_insert(uint32_t src, uint32_t dst, uint32_t timestamp,
                                     EdgeRelation relation, float weight) {
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

        // Live edges only. Expired prefixes are deliberately not counted, even
        // though they physically hold slots right now, because redistribute()
        // drops them -- so what has to fit in this window is the survivors, not
        // the garbage.
        //
        // Counting them would invert the intent: a window is escalated when it
        // is too dense, and escalating past a window that is mostly expired
        // walks all the way to the top and resizes the PMA, which is precisely
        // the wrong move when a single rewrite would have freed everything. A
        // sliding window over 16 vertices doubled its array twice under that
        // reading; counting live only, it never grows at all.
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
            redistribute(v_start, v_end, src, dst, timestamp, relation, weight);
            return;
        }

        if (full_array) [[unlikely]] {
            resize_pma(src);
            // retry against the doubled PMA
            insert_edge(src, dst, timestamp, relation, weight);
            return;
        }
    }
}

void PCSRGraph::redistribute(uint32_t v_start, uint32_t v_end,
                             uint32_t src, uint32_t dst, uint32_t timestamp,
                             EdgeRelation relation, float weight) {
    const uint32_t win_st = vertex_offsets[v_start];
    const uint32_t win_end = vertex_offsets[v_end + 1];
    const uint32_t win_capacity = win_end - win_st;
    const uint32_t n = v_end - v_start + 1;

    ++rebalance_count;
    slots_rewritten += win_capacity;

    // Pass 1: lift every live edge in the window into the scratchpad, in
    // vertex order, splicing the new edge into src's run as we pass it.
    //
    // Expired prefixes are skipped rather than copied, so this pass is also
    // where expiry actually returns memory. Nothing extra is done for it: the
    // dead slots simply are not carried across, and pass 2 lays the survivors
    // down from the window start with vertex_starts reset to zero.
    // Hoisted so the copy loop does not re-test it per slot.
    const bool weighted = edge_weights != nullptr;

    uint32_t scratch_idx = 0;
    uint64_t reclaimed = 0;
    for (uint32_t v = v_start; v <= v_end; ++v) {
        const uint32_t live_start = vertex_offsets[v] + vertex_starts[v];
        uint32_t count = vertex_counts[v];
        reclaimed += vertex_starts[v];

        for (uint32_t i = 0; i < count; ++i) {
            // Relations and weights move in lockstep with their edges; the
            // arrays are only meaningful while their indices agree.
            scratchpad_relations[scratch_idx] = edge_relations[live_start + i];
            if (weighted) scratchpad_weights[scratch_idx] = edge_weights[live_start + i];
            scratchpad_edges[scratch_idx++] = edges[live_start + i];
        }
        if (v == src) {
            scratchpad_relations[scratch_idx] = relation;
            if (weighted) scratchpad_weights[scratch_idx] = weight;
            scratchpad_edges[scratch_idx++] = {dst, timestamp};
            ++count;
        }
        scratchpad_counts[v - v_start] = count;
    }
    dead_slots -= reclaimed;

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
        vertex_starts[v_start + i] = 0;  // dead prefix reclaimed by this pass
        for (uint32_t j = 0; j < count; ++j) {
            edges[cursor + j] = scratchpad_edges[consumed];
            edge_relations[cursor + j] = scratchpad_relations[consumed];
            if (weighted) edge_weights[cursor + j] = scratchpad_weights[consumed];
            ++consumed;
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
    EdgeRelation* new_relations = static_cast<EdgeRelation*>(
        arena.allocate(static_cast<size_t>(new_capacity) * sizeof(EdgeRelation)));
    EdgeRelation* new_scratch_relations = static_cast<EdgeRelation*>(
        arena.allocate(static_cast<size_t>(new_capacity) * sizeof(EdgeRelation)));
    const bool weighted = edge_weights != nullptr;
    float* new_weights = weighted ? static_cast<float*>(
        arena.allocate(static_cast<size_t>(new_capacity) * sizeof(float))) : nullptr;
    float* new_scratch_weights = weighted ? static_cast<float*>(
        arena.allocate(static_cast<size_t>(new_capacity) * sizeof(float))) : nullptr;
    // A fresh offsets array: the old one has to stay readable while we walk it.
    uint32_t* new_offsets = static_cast<uint32_t*>(
        arena.allocate((static_cast<size_t>(num_vertices) + 1) * sizeof(uint32_t)));

    fill_gaps(new_edges, new_capacity);
    std::memset(new_relations, 0,
                static_cast<size_t>(new_capacity) * sizeof(EdgeRelation));
    if (weighted) {
        std::memset(new_weights, 0, static_cast<size_t>(new_capacity) * sizeof(float));
    }

    const uint32_t spare = new_capacity - static_cast<uint32_t>(num_edges);
    distribute_gaps(vertex_counts, num_vertices, num_edges, spare, scratchpad_gaps, hot);

    // Growth compacts too: only live edges cross to the new array, so any
    // expired prefix is dropped here as well.
    uint32_t cursor = 0;
    for (uint32_t v = 0; v < num_vertices; ++v) {
        const uint32_t live_start = vertex_offsets[v] + vertex_starts[v];
        const uint32_t count = vertex_counts[v];

        new_offsets[v] = cursor;
        for (uint32_t i = 0; i < count; ++i) {
            new_edges[cursor + i] = edges[live_start + i];
            new_relations[cursor + i] = edge_relations[live_start + i];
            if (weighted) new_weights[cursor + i] = edge_weights[live_start + i];
        }
        vertex_starts[v] = 0;
        cursor += count + scratchpad_gaps[v];
    }
    new_offsets[num_vertices] = new_capacity;
    dead_slots = 0;

    edges = new_edges;
    edge_relations = new_relations;
    edge_weights = new_weights;
    scratchpad_edges = new_scratch;
    scratchpad_relations = new_scratch_relations;
    scratchpad_weights = new_scratch_weights;
    vertex_offsets = new_offsets;
    edge_capacity = new_capacity;
}

uint32_t PCSRGraph::expire_vertex_before(uint32_t vertex, uint32_t timestamp) {
    if (vertex >= num_vertices) {
        throw std::out_of_range("requested node is nonexistent. nodes go from 0 to " +
                                std::to_string(num_vertices - 1) + ".");
    }

    const TemporalEdge* run = edges + vertex_offsets[vertex] + vertex_starts[vertex];
    const uint32_t count = vertex_counts[vertex];

    // Linear from the front rather than a bisection. The scan costs O(edges
    // actually removed), which beats O(log degree) in the case this exists for
    // -- a sliding window trimmed often, where each sweep drops a handful of
    // edges per vertex -- and matches it when the whole run goes.
    uint32_t removed = 0;
    while (removed < count && run[removed].timestamp < timestamp) {
        ++removed;
    }
    if (removed == 0) {
        return 0;
    }

    vertex_starts[vertex] += removed;
    vertex_counts[vertex] = count - removed;
    num_edges -= removed;
    expired_edges += removed;
    dead_slots += removed;
    return removed;
}

uint64_t PCSRGraph::expire_before(uint32_t timestamp) {
    uint64_t removed = 0;
    for (uint32_t v = 0; v < num_vertices; ++v) {
        removed += expire_vertex_before(v, timestamp);
    }
    return removed;
}
