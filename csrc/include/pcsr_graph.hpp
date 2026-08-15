#pragma once
#include <cstdint>
#include <cstddef>
#include "types.hpp"

class PCSRGraph {
    private:
        uint32_t num_vertices; 
        uint32_t edge_capacity; 
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
         */
        PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity);
        ~PCSRGraph();

        /**
         * @brief Inserts a dynamic temporal directed edge into the graph.
         * 
         * @param src Source node ID.
         * @param dst Target node ID.
         * @param timestamp Unix timestamp of the event.
         */
        void insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp);

        /**
         * @brief Returns raw memory pointer to vertex offsets for zero-copy numpy bindings.
         */
        const uint32_t* get_vertex_offsets() const { return vertex_offsets; }

        /**
         * @brief Returns raw memory pointer to edges array for zero-copy numpy bindings.
         */
        const TemporalEdge* get_edges() const { return edges; }
};