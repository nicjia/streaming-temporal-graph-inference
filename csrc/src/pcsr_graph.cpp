#pragma once
#include "pcsr_graph.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>


// structure is like edges array 
// for node i, the edges corresponding OUT from node i are located 
// at edges[vertex_offsets[i] to edges[vertex_offsets[i+1]]], 
// note that it goes up to i+1

static inline size_t pad_to_64(size_t size) {
    return (size + 63) & ~63; // Bitwise magic to snap to 64-byte boundaries
}

PCSRGraph::PCSRGraph(uint32_t max_vertices, uint32_t initial_edge_capacity)
    : num_vertices(max_vertices), edge_capacity(initial_edge_capacity) {

    vertex_offsets = static_cast<uint32_t*>(
        std::aligned_alloc(64, pad_to_64((num_vertices + 1) * sizeof(uint32_t))));
    edges = static_cast<TemporalEdge*>(
        std::aligned_alloc(64, pad_to_64(edge_capacity * sizeof(TemporalEdge))));

    scratchpad_edges = static_cast<TemporalEdge*>(std::aligned_alloc(64, pad_to_64(edge_capacity * sizeof(TemporalEdge))));
    scratchpad_counts = static_cast<uint32_t*>(std::aligned_alloc(64, pad_to_64(num_vertices * sizeof(uint32_t))));

    if (!vertex_offsets || !edges || !scratchpad_edges || !scratchpad_counts) {
        throw std::runtime_error("CRITICAL: std::aligned_alloc returned nullptr. Out of RAM or invalid alignment.");
    }

    for(uint32_t i = 0 ;i < edge_capacity; i++){
        edges[i] = {EMPTY_GAP,0};
    }
    uint32_t epv = edge_capacity/num_vertices;
    for(uint32_t i = 0; i <= num_vertices; i++){
        vertex_offsets[i] = i * epv;
    }
}

PCSRGraph::~PCSRGraph() {
    std::free(vertex_offsets);
    std::free(edges);
    std::free(scratchpad_edges);
    std::free(scratchpad_counts);
}

void PCSRGraph::insert_edge(uint32_t src, uint32_t dst, uint32_t timestamp){
    if(src >= num_vertices || dst >= num_vertices){
        throw std::out_of_range("requested node is nonexistent. nodes go from 0 to " + std::to_string(num_vertices-1) + ".");
    }

    uint32_t st = vertex_offsets[src];
    uint32_t end = vertex_offsets[src + 1];

    for(uint32_t i = st ;i < end; i++){
        if(edges[i].target_node == EMPTY_GAP){
            edges[i] = {dst, timestamp};
            return;
        }
    }
    rebalance_and_insert(src, dst, timestamp);
}

void PCSRGraph::rebalance_and_insert(uint32_t src, uint32_t dst, uint32_t timestamp){
    uint32_t v_start = src;
    uint32_t v_end = src;
    bool gap_found = false;

    while (!gap_found) {
        if (v_start > 0) v_start--;
        if (v_end < num_vertices - 1) v_end++;

        uint32_t win_st = vertex_offsets[v_start];
        uint32_t win_end = vertex_offsets[v_end + 1];

        for (uint32_t i = win_st; i < win_end; ++i) {
            if (edges[i].target_node == EMPTY_GAP) {
                gap_found = true;
                break;
            }
        }

        if (v_start == 0 && v_end == num_vertices - 1 && !gap_found) [[unlikely]] {
            resize_pma();
            insert_edge(src, dst, timestamp); // Retry after global double
            return;
        }
    }

    uint32_t win_st = vertex_offsets[v_start];
    uint32_t win_end = vertex_offsets[v_end + 1];
    uint32_t win_capacity = win_end - win_st;
    uint32_t num_nodes_in_win = v_end - v_start + 1;

    for (uint32_t i = 0; i < num_nodes_in_win; ++i) {
        scratchpad_counts[i] = 0;
    }

    uint32_t scratch_idx = 0;
    for (uint32_t v = v_start; v <= v_end; ++v) {
        uint32_t cur_st = vertex_offsets[v];
        uint32_t cur_end = vertex_offsets[v + 1];

        for (uint32_t i = cur_st; i < cur_end; ++i) {
            if (edges[i].target_node != EMPTY_GAP) {
                scratchpad_edges[scratch_idx++] = edges[i];
                scratchpad_counts[v - v_start]++;
            }
        }
        if (v == src) {
            scratchpad_edges[scratch_idx++] = {dst, timestamp};
            scratchpad_counts[v - v_start]++;
        }
    }

    for (uint32_t i = win_st; i < win_end; ++i) {
        edges[i] = {EMPTY_GAP, 0};
    }

    uint32_t slots_per_node = win_capacity / num_nodes_in_win;
    uint32_t current_main = win_st;
    uint32_t current_scratch = 0;

    for (uint32_t i = 0; i < num_nodes_in_win; ++i) {
        vertex_offsets[v_start + i] = current_main;
        uint32_t edges_for_this_node = scratchpad_counts[i];
        
        for (uint32_t j = 0; j < edges_for_this_node; ++j) {
            edges[current_main + j] = scratchpad_edges[current_scratch++];
        }
        current_main += slots_per_node;
    }
    vertex_offsets[v_end + 1] = win_end;
}

void PCSRGraph::resize_pma(){
    uint32_t new_capacity = edge_capacity * 2;
    
    TemporalEdge* new_edges = static_cast<TemporalEdge*>(std::aligned_alloc(64, new_capacity * sizeof(TemporalEdge)));
    TemporalEdge* new_scratch = static_cast<TemporalEdge*>(std::aligned_alloc(64, new_capacity * sizeof(TemporalEdge)));
    
    for (uint32_t i = 0; i < new_capacity; ++i) {
        new_edges[i] = {EMPTY_GAP, 0};
    }

    uint32_t slots_per_vertex = new_capacity / num_vertices;
    uint32_t current_ptr = 0;

    for (uint32_t v = 0; v < num_vertices; ++v) {
        uint32_t old_st = vertex_offsets[v];
        uint32_t old_end = vertex_offsets[v + 1];
        vertex_offsets[v] = current_ptr;
        
        for (uint32_t i = old_st; i < old_end; ++i) {
            if (edges[i].target_node != EMPTY_GAP) {
                new_edges[current_ptr++] = edges[i];
            }
        }
        current_ptr = vertex_offsets[v] + slots_per_vertex;
    }
    vertex_offsets[num_vertices] = new_capacity;

    std::free(edges);
    std::free(scratchpad_edges);
    
    edges = new_edges;
    scratchpad_edges = new_scratch;
    edge_capacity = new_capacity;
}