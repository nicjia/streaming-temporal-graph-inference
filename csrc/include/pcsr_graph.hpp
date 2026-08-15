#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"
#include "memory_arena.hpp"

class PCSRGraph {
    private:
        uint32_t num_vertices; 
        uint32_t edge_capacity; 

        MemoryArena arena;

        alignas(64) uint32_t* vertex_offsets; //size = num_vertices + 1
        alignas(64) TemporalEdge* edges; //size = edge_capacity

        alignas(64) TemporalEdge* scratchpad_edges;
        alignas(64) uint32_t* scratchpad_counts; 

        void rebalance_and_insert(uint32_t src, uint32_t dst, uint32_t timestamp);
        void resize_pma();
    public:
        /**
         * @brief Constructs a PCSR Graph instance.
         * @param max_vertices Maximum number of nodes in the graph (V).
         * @param initial_edge_capacity Initial size of the Packed Memory Array (E + Gaps).
         * @param arena_bytes Size of the memory arena to use for the graph.
         */
        PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity, size_t arena_bytes = 128 * 1024 * 1024);
        ~PCSRGraph();

        void insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp);
        const uint32_t* get_vertex_offsets() const { return vertex_offsets; }
        const TemporalEdge* get_edges() const { return edges; }
        uint32_t get_num_vertices() const { return num_vertices; }
        uint32_t get_edge_capacity() const { return edge_capacity; }
};