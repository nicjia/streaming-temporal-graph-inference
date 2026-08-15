#include "pcsr_graph.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

int main() {
    const uint32_t NUM_NODES = 100'000;
    const uint32_t NUM_EDGES = 10'000'000;
    
    std::cout << "Generating " << NUM_EDGES << " random edges...\n";
    
    std::vector<uint32_t> srcs(NUM_EDGES);
    std::vector<uint32_t> dsts(NUM_EDGES);
    std::vector<uint32_t> timestamps(NUM_EDGES);

    // Random number generation for realistic scatter
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist_node(0, NUM_NODES - 1);
    std::uniform_int_distribution<uint32_t> dist_ts(1000000, 2000000);

    for (uint32_t i = 0; i < NUM_EDGES; ++i) {
        srcs[i] = dist_node(rng);
        dsts[i] = dist_node(rng);
        timestamps[i] = dist_ts(rng);
    }

    std::cout << "Initializing PCSRGraph (100k nodes, 15M capacity)...\n";
    // 500MB arena allocation to comfortably hold everything without hitting OS limits
    PCSRGraph graph(NUM_NODES, NUM_EDGES + 5'000'000, 500 * 1024 * 1024);

    std::cout << "Starting pure C++ insertion stress test...\n";
    
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < NUM_EDGES; ++i) {
        graph.insert_edge(srcs[i], dsts[i], timestamps[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    double seconds = diff.count();
    double throughput = NUM_EDGES / seconds;

    std::cout << "======================================\n";
    std::cout << "Total Edges Inserted : " << NUM_EDGES << "\n";
    std::cout << "Total Time Taken     : " << std::fixed << std::setprecision(4) << seconds << " seconds\n";
    std::cout << "C++ Throughput       : " << std::fixed << std::setprecision(0) << throughput << " inserts/second\n";
    std::cout << "======================================\n";

    return 0;
}