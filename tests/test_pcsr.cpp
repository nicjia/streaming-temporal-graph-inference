#include "pcsr_graph.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running PCSRGraph Unit Test...\n";

    // 10 nodes, 100 capacity, 10MB arena
    PCSRGraph graph(10, 100, 10 * 1024 * 1024);

    // Insert a few edges
    graph.insert_edge(0, 5, 1000);
    graph.insert_edge(0, 6, 1001);
    graph.insert_edge(1, 2, 1002);

    const uint32_t* offsets = graph.get_vertex_offsets();
    const TemporalEdge* edges = graph.get_edges();

    // Verify Node 0's edges are sitting in its local window
    uint32_t start_0 = offsets[0];
    
    // We expect the first valid edge for node 0 to be {target: 5, timestamp: 1000}
    assert(edges[start_0].target_node == 5);
    assert(edges[start_0].timestamp == 1000);

    // Second edge
    assert(edges[start_0 + 1].target_node == 6);
    assert(edges[start_0 + 1].timestamp == 1001);

    std::cout << "All assertions passed! The graph memory and insert logic work perfectly.\n";
    return 0;
}