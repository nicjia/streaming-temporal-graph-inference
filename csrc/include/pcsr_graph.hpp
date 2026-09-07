#pragma once
#include <atomic>
#include <cstdint>
#include <span>
#include <cstddef>
#include "types.hpp"
#include "memory_arena.hpp"
#include <stdexcept>

/**
 * @brief Packed Compressed Sparse Row graph over a Packed Memory Array.
 *
 * Layout: one flat `edges` array carved into a contiguous region per vertex.
 * Vertex v owns slots [vertex_offsets[v], vertex_offsets[v+1]). Within that
 * region the first vertex_starts[v] slots are expired, the next
 * vertex_counts[v] hold live edges, and the rest are gaps marked with
 * EMPTY_GAP. Keeping every region left-packed makes insertion O(1) (write at
 * offset+start+count) instead of a linear scan for a free slot, and makes a
 * vertex's adjacency list a single unbroken run for readers.
 *
 * vertex_starts is zero on a graph that has never expired anything, which is
 * the case every reader saw before expiry existed. See expire_before().
 *
 * When a region fills, rebalance_and_insert() rewrites a power-of-two window of
 * vertices around the hot one, redistributing the window's free slots in
 * proportion to each vertex's degree. Windows double until one is under its
 * density bound, which is what buys the O(log^2 N) amortized bound; if even the
 * whole array is too dense, resize_pma() doubles capacity.
 *
 * The arrays are 64-byte aligned because MemoryArena aligns every block. Note
 * that `alignas(64)` on a pointer member would align the 8-byte pointer, not
 * the array it points at -- the guarantee has to come from the allocator.
 */
class PCSRGraph {
    private:
        uint32_t num_vertices;
        uint32_t edge_capacity;
        uint64_t num_edges; // live edges across the whole graph

        // Instrumentation. Cheap enough to leave on: the counters are only
        // touched on the slow path, and they are how you tell a healthy PMA
        // from one that is thrashing.
        uint64_t rebalance_count;
        uint64_t resize_count;
        uint64_t slots_rewritten;
        uint64_t expired_edges;   // lifetime total removed by expire_*
        uint64_t dead_slots;      // expired slots not yet reclaimed by a rebalance

        // Single-writer enforcement. See WriteGuard.
        std::atomic<bool> writer_active;

        MemoryArena arena;

        uint32_t* vertex_offsets;  // size = num_vertices + 1
        uint32_t* vertex_counts;   // size = num_vertices, live edges per vertex
        uint32_t* vertex_starts;   // size = num_vertices, expired prefix per region
        TemporalEdge* edges;       // size = edge_capacity
        EdgeRelation* edge_relations;  // size = edge_capacity, parallel to edges
        float* edge_weights;           // size = edge_capacity, or null -- see below

        TemporalEdge* scratchpad_edges;  // size = edge_capacity
        EdgeRelation* scratchpad_relations;  // size = edge_capacity
        float* scratchpad_weights;       // size = edge_capacity, or null
        uint32_t* scratchpad_counts;     // size = num_vertices
        uint32_t* scratchpad_gaps;       // size = num_vertices

        void rebalance_and_insert(uint32_t src, uint32_t dst, uint32_t timestamp,
                                  EdgeRelation relation, float weight);
        void redistribute(uint32_t v_start, uint32_t v_end,
                          uint32_t src, uint32_t dst, uint32_t timestamp,
                          EdgeRelation relation, float weight);
        void resize_pma(uint32_t hot);
    public:
        /**
         * @brief Asserts exclusive write access for its lifetime.
         *
         * PCSRGraph is single-writer: a rebalance rewrites whole windows of the
         * edge array and reseats vertex offsets, so two concurrent mutators
         * interleave into corruption rather than merely racing on a counter.
         *
         * This matters specifically because the Python bulk-insert path
         * releases the GIL to get its throughput. Before that, the interpreter
         * lock serialised every call for free; afterwards two Python threads
         * could sit inside the same graph's insert loop at once. Measured, that
         * lost roughly two thirds of the edges and left the reported edge count
         * disagreeing with a full scan -- silently, with no crash.
         *
         * Throwing rather than locking is deliberate: concurrent mutation is a
         * programming error here, not a case to serialise transparently, and a
         * mutex would hide the mistake while halving the throughput the GIL
         * release was bought for. Readers are unaffected; a graph with no
         * active writer can be read from any number of threads.
         */
        class WriteGuard {
            PCSRGraph* graph;
        public:
            explicit WriteGuard(PCSRGraph& target) : graph(&target) {
                bool expected = false;
                if (!graph->writer_active.compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel)) {
                    graph = nullptr;  // do not release a lock we never took
                    throw std::runtime_error(
                        "PCSRGraph is single-writer: another thread is already "
                        "inserting into this graph. Serialise your writers, or "
                        "give each thread its own graph.");
                }
            }
            ~WriteGuard() {
                if (graph) {
                    graph->writer_active.store(false, std::memory_order_release);
                }
            }
            WriteGuard(const WriteGuard&) = delete;
            WriteGuard& operator=(const WriteGuard&) = delete;
        };

        /**
         * @brief Constructs a PCSR Graph instance.
         * @param max_vertices Maximum number of nodes in the graph (V). Must be > 0.
         * @param initial_edge_capacity Initial size of the Packed Memory Array (E + Gaps).
         *        Clamped up to max_vertices so every vertex gets at least one slot.
         * @param arena_bytes Size of the memory arena to use for the graph. A slot
         *        costs 28 bytes across the edge, relation and weight arrays plus
         *        their scratchpads. Growing the PMA re-allocates from this arena
         *        without reclaiming the old blocks, so budget roughly 64 bytes per
         *        final edge slot if you rely on growth.
         * @param store_weights Allocate the per-edge float weight array. Off by
         *        default, because it is not free: the struct-of-arrays layout that
         *        makes a parallel array cheap to *read* makes it expensive to
         *        *write*, since every insert then touches a third cache line.
         *        Measured at 35.1 -> 54.1 ns/edge on a 10M-edge uniform load that
         *        rebalances exactly once, so that 1.54x is the fast path alone,
         *        plus 40% more memory. Workloads that carry a size or volume want
         *        it; the rest should not pay for it. With this off, insert_edge
         *        throws rather than silently discarding a non-zero weight.
         */
        PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity,
                  size_t arena_bytes = 128 * 1024 * 1024, bool store_weights = false);
        ~PCSRGraph();

        void insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp,
                         EdgeRelation relation = RELATION_UNKNOWN,
                         float weight = 0.0f);

        /**
         * @brief Drops the leading run of each adjacency older than `timestamp`.
         *
         * Bounded-memory streaming. Without this the PMA only grows, so a live
         * feed eventually exhausts the arena no matter how much history the
         * model actually reads.
         *
         * The implementation is a prefix removal, which is the cheapest case a
         * PMA admits and is available here only because regions are both
         * left-packed and chronological. There is no tombstoning and no
         * compaction pass: vertex_starts[v] advances past the dead edges and
         * vertex_counts[v] shrinks by the same amount, so the cost is O(V +
         * edges actually removed) with no memory movement at all. The dead
         * prefix stays allocated until the next rebalance touches that window,
         * which reclaims it for free -- redistribute() copies only live edges.
         *
         * Because the dead prefix still occupies slots, it counts toward the
         * density bound until reclaimed. That is deliberate: it is what makes a
         * heavily-expired window rebalance itself and hand the space back.
         *
         * Semantics are exactly "remove the longest prefix of each adjacency
         * whose timestamps are < timestamp". For a graph built by chronological
         * insertion -- the assumption the sampler already makes -- that is the
         * same as "remove every edge older than timestamp". For one built out
         * of order it removes a prefix and stops at the first surviving edge,
         * which keeps the run contiguous rather than punching holes in it.
         *
         * @return Number of edges removed.
         */
        uint64_t expire_before(uint32_t timestamp);

        /// expire_before() restricted to one vertex. O(edges removed).
        uint32_t expire_vertex_before(uint32_t vertex, uint32_t timestamp);

        /**
         * @brief The live out-edges of a vertex, as a contiguous span.
         *
         * Prefer this over indexing get_edges() with get_vertex_offsets(). The
         * obvious hand-written loop
         *
         *     for (i = 0; i < counts[v]; ++i) sum += edges[offsets[v] + i]...
         *
         * runs at less than half speed, because the compiler cannot prove the
         * edge array does not alias the counts array and so reloads counts[v]
         * on every iteration. Measured at 4.4 GB/s against 10.0 GB/s for the
         * identical loop with the bound hoisted into a local. Handing back a
         * span makes the fast form the natural one.
         *
         * Regions are left-packed, so the span is gap-free and contains
         * exactly the live edges in insertion order.
         */
        std::span<const TemporalEdge> neighbors(uint32_t v) const {
            return {edges + vertex_offsets[v] + vertex_starts[v], vertex_counts[v]};
        }

        /// Relation codes parallel to neighbors(v), same indices.
        std::span<const EdgeRelation> neighbor_relations(uint32_t v) const {
            return {edge_relations + vertex_offsets[v] + vertex_starts[v], vertex_counts[v]};
        }

        /// Edge weights parallel to neighbors(v), same indices. Empty unless the
        /// graph was constructed with store_weights.
        std::span<const float> neighbor_weights(uint32_t v) const {
            if (!edge_weights) return {};
            return {edge_weights + vertex_offsets[v] + vertex_starts[v], vertex_counts[v]};
        }

        /// True if this graph carries per-edge weights.
        bool has_weights() const { return edge_weights != nullptr; }

        const uint32_t* get_vertex_offsets() const { return vertex_offsets; }
        const uint32_t* get_vertex_counts() const { return vertex_counts; }
        /// Expired prefix per region. All zero unless expire_* has been called.
        const uint32_t* get_vertex_starts() const { return vertex_starts; }
        const TemporalEdge* get_edges() const { return edges; }
        const EdgeRelation* get_edge_relations() const { return edge_relations; }
        const float* get_edge_weights() const { return edge_weights; }
        uint32_t get_num_vertices() const { return num_vertices; }
        uint32_t get_edge_capacity() const { return edge_capacity; }
        uint64_t get_num_edges() const { return num_edges; }
        uint64_t get_rebalance_count() const { return rebalance_count; }
        uint64_t get_resize_count() const { return resize_count; }
        /// Total slots touched by rebalances and growth -- the real work done
        /// beyond the O(1) fast path.
        uint64_t get_slots_rewritten() const { return slots_rewritten; }
        /// Lifetime edges removed by expire_*.
        uint64_t get_expired_edges() const { return expired_edges; }
        /// Expired slots still occupying the PMA, awaiting the next rebalance.
        uint64_t get_dead_slots() const { return dead_slots; }
        uint32_t get_degree(uint32_t v) const { return vertex_counts[v]; }
        size_t get_arena_used() const { return arena.get_used(); }
};
