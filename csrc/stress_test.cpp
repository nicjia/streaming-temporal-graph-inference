#include "pcsr_graph.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

// Helper structure for pre-generating workloads
struct EdgeRequest {
    uint32_t src;
    uint32_t dst;
    uint32_t timestamp;
};

// Benchmark harness
void run_benchmark(const std::string& name, PCSRGraph& graph, const std::vector<EdgeRequest>& edges) {
    std::cout << "[Running] " << name << " (" << edges.size() << " edges)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& edge : edges) {
        graph.insert_edge(edge.src, edge.dst, edge.timestamp);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    double seconds = diff.count();
    double edges_per_sec = edges.size() / seconds;

    std::cout << "  -> Time:       " << std::fixed << std::setprecision(4) << seconds << " seconds\n";
    std::cout << "  -> Throughput: " << std::fixed << std::setprecision(0) << edges_per_sec << " edges/sec\n\n";
}

int main() {
    const uint32_t NUM_VERTICES = 1'000'000;
    const uint32_t INITIAL_CAPACITY = 5'000'000;
    const uint32_t NUM_INSERTS = 3'000'000;

    std::mt19937 rng(42); // Fixed seed for reproducibility
    
    // =========================================================================
    // Use Case 1: Sequential Insertion (Best Case - Zero Rebalancing)
    // =========================================================================
    {
        PCSRGraph graph(NUM_VERTICES, INITIAL_CAPACITY);
        std::vector<EdgeRequest> workload(NUM_INSERTS);
        
        for (uint32_t i = 0; i < NUM_INSERTS; ++i) {
            workload[i] = {i % NUM_VERTICES, (i + 1) % NUM_VERTICES, i};
        }
        
        run_benchmark("Sequential Insertion (Perfect Distribution)", graph, workload);
    }

    // =========================================================================
    // Use Case 2: Uniform Random Distribution (Real-world Baseline)
    // =========================================================================
    {
        PCSRGraph graph(NUM_VERTICES, INITIAL_CAPACITY);
        std::vector<EdgeRequest> workload(NUM_INSERTS);
        std::uniform_int_distribution<uint32_t> dist(0, NUM_VERTICES - 1);
        
        for (uint32_t i = 0; i < NUM_INSERTS; ++i) {
            workload[i] = {dist(rng), dist(rng), i};
        }
        
        run_benchmark("Uniform Random Graph (Standard Rebalancing)", graph, workload);
    }

    // =========================================================================
    // Use Case 3: The "Super-Node" Hotspot (Worst Case - Heavy Local Shifts)
    // =========================================================================
    // Simulates an influencer on Twitter or a central routing hub.
    // 90% of edges go to node 0, causing massive cascading local window rebalances.
    {
        PCSRGraph graph(NUM_VERTICES, INITIAL_CAPACITY);
        std::vector<EdgeRequest> workload(NUM_INSERTS);
        std::uniform_int_distribution<uint32_t> dist_all(0, NUM_VERTICES - 1);
        std::uniform_int_distribution<int> chance(1, 100);
        
        for (uint32_t i = 0; i < NUM_INSERTS; ++i) {
            uint32_t src = (chance(rng) <= 90) ? 0 : dist_all(rng);
            workload[i] = {src, dist_all(rng), i};
        }
        
        run_benchmark("Super-Node Hotspot (90% traffic to Node 0)", graph, workload);
    }

    // =========================================================================
    // Use Case 4: Global Resize Panic (Forcing Dynamic Allocations)
    // =========================================================================
    // Start with a tiny capacity to force multiple Stop-The-World global resizes.
    {
        PCSRGraph graph(NUM_VERTICES, 1'000'000); // Only 1M capacity
        std::vector<EdgeRequest> workload(NUM_INSERTS); 
        std::uniform_int_distribution<uint32_t> dist(0, NUM_VERTICES - 1);
        
        // Inserting 3M edges into 1M capacity guarantees at least 2 global resizes
        for (uint32_t i = 0; i < NUM_INSERTS; ++i) {
            workload[i] = {dist(rng), dist(rng), i};
        }
        
        run_benchmark("Global Resize Panic (Forcing PMA Capacity Doubling)", graph, workload);
    }

    return 0;
}