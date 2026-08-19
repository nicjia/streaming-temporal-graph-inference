#include "pcsr_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Workload {
    std::string name;
    std::vector<uint32_t> src;
    std::vector<uint32_t> dst;
    std::vector<uint32_t> ts;
};

/**
 * Uniform endpoints. This is the easy case and it is NOT representative: with
 * capacity/vertices slots per region and a matching mean degree, almost every
 * insert lands on the O(1) fast path and the PMA rebalance logic barely runs.
 * Kept only as a ceiling to compare the realistic workload against.
 */
Workload make_uniform(uint32_t vertices, uint32_t edges, uint32_t seed) {
    Workload w{"uniform", {}, {}, {}};
    w.src.resize(edges); w.dst.resize(edges); w.ts.resize(edges);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> node(0, vertices - 1);
    std::uniform_int_distribution<uint32_t> stamp(1'000'000, 2'000'000);

    for (uint32_t i = 0; i < edges; ++i) {
        w.src[i] = node(rng);
        w.dst[i] = node(rng);
        w.ts[i] = stamp(rng);
    }
    return w;
}

/**
 * Zipf-distributed sources: a few vertices carry most of the edges, which is
 * what GDELT actually looks like (UNITED STATES appears ~90x more often than
 * the median actor in a single 15-minute slice). This is the workload that
 * drives regions past their initial size and forces real rebalancing.
 */
Workload make_zipf(uint32_t vertices, uint32_t edges, double alpha, uint32_t seed) {
    Workload w{"zipf(alpha=" + std::to_string(alpha).substr(0, 4) + ")", {}, {}, {}};
    w.src.resize(edges); w.dst.resize(edges); w.ts.resize(edges);

    // Cumulative distribution over rank r: P(r) proportional to 1/r^alpha.
    std::vector<double> cdf(vertices);
    double running = 0.0;
    for (uint32_t r = 0; r < vertices; ++r) {
        running += 1.0 / std::pow(static_cast<double>(r + 1), alpha);
        cdf[r] = running;
    }
    for (uint32_t r = 0; r < vertices; ++r) {
        cdf[r] /= running;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> node(0, vertices - 1);
    std::uniform_int_distribution<uint32_t> stamp(1'000'000, 2'000'000);

    for (uint32_t i = 0; i < edges; ++i) {
        const double u = unit(rng);
        w.src[i] = static_cast<uint32_t>(
            std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        if (w.src[i] >= vertices) w.src[i] = vertices - 1;
        w.dst[i] = node(rng);
        w.ts[i] = stamp(rng);
    }
    return w;
}

void describe_skew(const Workload& w, uint32_t vertices) {
    std::vector<uint32_t> degree(vertices, 0);
    for (uint32_t s : w.src) ++degree[s];
    std::sort(degree.begin(), degree.end(), std::greater<uint32_t>());

    const uint32_t top = std::max<uint32_t>(1, vertices / 100);
    uint64_t top_sum = 0;
    for (uint32_t i = 0; i < top; ++i) top_sum += degree[i];

    std::cout << "  skew            : max degree " << degree[0]
              << ", median " << degree[vertices / 2]
              << ", top 1% hold " << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(top_sum) / w.src.size()) << "% of edges\n";
}

/// Walks every adjacency list, the way an ML feature extractor would.
uint64_t scan_all(const PCSRGraph& g, uint64_t& checksum) {
    const uint32_t* offsets = g.get_vertex_offsets();
    const uint32_t* counts = g.get_vertex_counts();
    const TemporalEdge* edges = g.get_edges();

    uint64_t seen = 0;
    uint64_t sum = 0;
    for (uint32_t v = 0; v < g.get_num_vertices(); ++v) {
        for (uint32_t i = 0; i < counts[v]; ++i) {
            sum += edges[offsets[v] + i].target_node;
            ++seen;
        }
    }
    checksum = sum;
    return seen;
}

/// Baseline cost of two clock reads, so per-insert percentiles stay honest.
double clock_overhead_ns() {
    const int samples = 100000;
    std::vector<double> deltas(samples);
    for (int i = 0; i < samples; ++i) {
        const auto a = Clock::now();
        const auto b = Clock::now();
        deltas[i] = std::chrono::duration<double, std::nano>(b - a).count();
    }
    std::sort(deltas.begin(), deltas.end());
    return deltas[samples / 2];
}

void run(const Workload& w, uint32_t vertices, uint32_t capacity, size_t arena_bytes,
         bool measure_percentiles, double overhead_ns) {
    std::cout << "\n--- " << w.name << " | " << vertices << " vertices, "
              << w.src.size() << " edges, " << capacity << " initial slots ---\n";
    describe_skew(w, vertices);

    PCSRGraph graph(vertices, capacity, arena_bytes);

    const auto start = Clock::now();
    for (size_t i = 0; i < w.src.size(); ++i) {
        graph.insert_edge(w.src[i], w.dst[i], w.ts[i]);
    }
    const auto end = Clock::now();

    const double seconds = std::chrono::duration<double>(end - start).count();
    const double throughput = static_cast<double>(w.src.size()) / seconds;

    uint64_t checksum = 0;
    const auto read_start = Clock::now();
    const uint64_t retained = scan_all(graph, checksum);
    const auto read_end = Clock::now();
    const double read_seconds = std::chrono::duration<double>(read_end - read_start).count();

    std::cout << "  insert          : " << std::fixed << std::setprecision(4) << seconds
              << " s  (" << std::setprecision(0) << throughput << " edges/s, "
              << std::setprecision(1) << (seconds * 1e9 / static_cast<double>(w.src.size()))
              << " ns/edge)\n";
    std::cout << "  full scan       : " << std::setprecision(4) << read_seconds << " s  ("
              << std::setprecision(0)
              << (static_cast<double>(retained) / read_seconds) << " edges/s read)\n";
    std::cout << "  retained        : " << retained << " / " << w.src.size();
    if (retained != w.src.size()) {
        std::cout << "   *** " << (w.src.size() - retained) << " EDGES LOST ***";
    } else {
        std::cout << "   (lossless)";
    }
    std::cout << "\n";
    std::cout << "  grew to         : " << graph.get_edge_capacity() << " slots, arena used "
              << std::setprecision(1)
              << (static_cast<double>(graph.get_arena_used()) / (1024 * 1024)) << " MB\n";
    std::cout << "  slow path       : " << graph.get_rebalance_count() << " rebalances, "
              << graph.get_resize_count() << " resizes, "
              << std::setprecision(2)
              << (static_cast<double>(graph.get_slots_rewritten()) /
                  static_cast<double>(w.src.size()))
              << " slots rewritten per edge inserted\n";
    std::cout << "  checksum        : " << checksum << "\n";

    if (!measure_percentiles) return;

    // Second pass, timing each insert individually. Tail latency is what a
    // streaming consumer actually feels when a rebalance fires.
    PCSRGraph fresh(vertices, capacity, arena_bytes);
    std::vector<double> latencies(w.src.size());
    for (size_t i = 0; i < w.src.size(); ++i) {
        const auto a = Clock::now();
        fresh.insert_edge(w.src[i], w.dst[i], w.ts[i]);
        const auto b = Clock::now();
        latencies[i] = std::chrono::duration<double, std::nano>(b - a).count() - overhead_ns;
    }
    std::sort(latencies.begin(), latencies.end());

    auto pct = [&](double p) { return latencies[static_cast<size_t>(p * (latencies.size() - 1))]; };
    std::cout << "  latency (ns)    : p50 " << std::setprecision(0) << std::max(0.0, pct(0.50))
              << "  p99 " << std::max(0.0, pct(0.99))
              << "  p99.9 " << std::max(0.0, pct(0.999))
              << "  max " << latencies.back() << "\n";
}

} // namespace

int main() {
    const uint32_t NUM_NODES = 100'000;
    const uint32_t NUM_EDGES = 10'000'000;

    std::cout << "PCSRGraph insertion benchmark\n";
    std::cout << "Generating workloads (" << NUM_EDGES << " edges each)...\n";

    const double overhead = clock_overhead_ns();
    std::cout << "Clock overhead (subtracted from percentiles): "
              << std::fixed << std::setprecision(1) << overhead << " ns\n";

    const Workload uniform = make_uniform(NUM_NODES, NUM_EDGES, 42);
    const Workload zipf = make_zipf(NUM_NODES, NUM_EDGES, 1.1, 42);

    // Pre-sized: 15M slots for 10M edges, the original benchmark's setup.
    run(uniform, NUM_NODES, NUM_EDGES + 5'000'000, 700ull * 1024 * 1024, false, overhead);
    run(zipf, NUM_NODES, NUM_EDGES + 5'000'000, 700ull * 1024 * 1024, false, overhead);

    // Under-sized: forces resize_pma() doublings on top of the skew. The arena
    // never reclaims the old blocks, hence the generous budget.
    run(zipf, NUM_NODES, 1'000'000, 2048ull * 1024 * 1024, false, overhead);

    // Percentiles on a smaller run so the double-buffered pass stays cheap.
    const Workload zipf_small = make_zipf(NUM_NODES, 1'000'000, 1.1, 7);
    run(zipf_small, NUM_NODES, 2'000'000, 256ull * 1024 * 1024, true, overhead);

    return 0;
}
