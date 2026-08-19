#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "memory_arena.hpp"

/**
 * @brief Packed Compressed Sparse Row graph over a Packed Memory Array.
 *
 * Layout: one flat `edges` array carved into a contiguous region per vertex.
 * Vertex v owns slots [vertex_offsets[v], vertex_offsets[v+1]); the first
 * vertex_counts[v] of them hold live edges and the rest are gaps marked with
 * EMPTY_GAP. Keeping every region left-packed makes insertion O(1) (write at
 * offset+count) instead of a linear scan for a free slot, and makes a vertex's
 * adjacency list a single unbroken run for readers.
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

        MemoryArena arena;

        uint32_t* vertex_offsets;  // size = num_vertices + 1
        uint32_t* vertex_counts;   // size = num_vertices, live edges per vertex
        TemporalEdge* edges;       // size = edge_capacity

        TemporalEdge* scratchpad_edges;  // size = edge_capacity
        uint32_t* scratchpad_counts;     // size = num_vertices
        uint32_t* scratchpad_gaps;       // size = num_vertices

        void rebalance_and_insert(uint32_t src, uint32_t dst, uint32_t timestamp);
        void redistribute(uint32_t v_start, uint32_t v_end,
                          uint32_t src, uint32_t dst, uint32_t timestamp);
        void resize_pma(uint32_t hot);
    public:
        /**
         * @brief Constructs a PCSR Graph instance.
         * @param max_vertices Maximum number of nodes in the graph (V). Must be > 0.
         * @param initial_edge_capacity Initial size of the Packed Memory Array (E + Gaps).
         *        Clamped up to max_vertices so every vertex gets at least one slot.
         * @param arena_bytes Size of the memory arena to use for the graph. Growing the
         *        PMA re-allocates from this arena without reclaiming the old blocks, so
         *        budget roughly 32 bytes per final edge slot if you rely on growth.
         */
        PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity, size_t arena_bytes = 128 * 1024 * 1024);
        ~PCSRGraph();

        void insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp);

        const uint32_t* get_vertex_offsets() const { return vertex_offsets; }
        const uint32_t* get_vertex_counts() const { return vertex_counts; }
        const TemporalEdge* get_edges() const { return edges; }
        uint32_t get_num_vertices() const { return num_vertices; }
        uint32_t get_edge_capacity() const { return edge_capacity; }
        uint64_t get_num_edges() const { return num_edges; }
        uint64_t get_rebalance_count() const { return rebalance_count; }
        uint64_t get_resize_count() const { return resize_count; }
        /// Total slots touched by rebalances and growth -- the real work done
        /// beyond the O(1) fast path.
        uint64_t get_slots_rewritten() const { return slots_rewritten; }
        uint32_t get_degree(uint32_t v) const { return vertex_counts[v]; }
        size_t get_arena_used() const { return arena.get_used(); }
};
