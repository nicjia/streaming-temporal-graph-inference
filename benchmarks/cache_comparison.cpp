// Does the layout actually buy what the design claims?
//
// The project asserts that a packed, cache-line-aligned edge array beats
// scattered heap allocation. Every benchmark so far has measured wall-clock
// insert throughput, which does not test that claim: a structure can be fast
// for reasons unrelated to locality. This one isolates it.
//
// Method, given no portable access to hardware counters: compare three layouts
// holding byte-identical logical data, and measure the *effective bandwidth* of
// a traversal -- useful bytes divided by elapsed time. A layout that reads
// contiguously approaches the machine's streaming bandwidth. A layout that
// chases pointers pulls a 64-byte line per access and uses 8 bytes of it, so
// its effective bandwidth collapses by roughly the ratio of line size to
// payload size. Sweeping the working set from L1 through DRAM makes the
// divergence appear exactly where the caches run out, which is the signature
// that distinguishes a locality effect from a constant-factor one.

#include "pcsr_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace {

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// ---------------------------------------------------------------------------
// Baseline 1: the obvious dynamic adjacency. One heap allocation per vertex,
// growing independently, so neighbouring vertices land arbitrarily far apart.
// ---------------------------------------------------------------------------
class NaiveAdjacency {
    std::vector<std::vector<TemporalEdge>> adjacency;
public:
    explicit NaiveAdjacency(uint32_t vertices) : adjacency(vertices) {}
    void insert(uint32_t src, uint32_t dst, uint32_t ts) {
        adjacency[src].push_back({dst, ts});
    }
    uint64_t scan_all() const {
        uint64_t sum = 0;
        for (const auto& run : adjacency) {
            for (const auto& e : run) sum += e.target_node;
        }
        return sum;
    }
    uint64_t scan_vertex(uint32_t v, uint32_t limit) const {
        uint64_t sum = 0;
        const auto& run = adjacency[v];
        const size_t take = std::min<size_t>(limit, run.size());
        for (size_t i = run.size() - take; i < run.size(); ++i) sum += run[i].target_node;
        return sum;
    }
};

// ---------------------------------------------------------------------------
// Baseline 2: hash map of vectors. Common in quick prototypes; adds a hash
// lookup and another indirection on top of the scattered allocation.
// ---------------------------------------------------------------------------
class HashAdjacency {
    std::unordered_map<uint32_t, std::vector<TemporalEdge>> adjacency;
public:
    explicit HashAdjacency(uint32_t vertices) { adjacency.reserve(vertices); }
    void insert(uint32_t src, uint32_t dst, uint32_t ts) {
        adjacency[src].push_back({dst, ts});
    }
    uint64_t scan_all() const {
        uint64_t sum = 0;
        for (const auto& [key, run] : adjacency) {
            (void)key;
            for (const auto& e : run) sum += e.target_node;
        }
        return sum;
    }
    uint64_t scan_vertex(uint32_t v, uint32_t limit) const {
        auto it = adjacency.find(v);
        if (it == adjacency.end()) return 0;
        uint64_t sum = 0;
        const auto& run = it->second;
        const size_t take = std::min<size_t>(limit, run.size());
        for (size_t i = run.size() - take; i < run.size(); ++i) sum += run[i].target_node;
        return sum;
    }
};

// ---------------------------------------------------------------------------
// Baseline 3: static CSR. The read-optimal layout, and the honest ceiling --
// perfectly packed, but immutable, so every batch costs a full rebuild. This
// is what PCSR is trying to match on reads while staying mutable.
// ---------------------------------------------------------------------------
class StaticCSR {
    std::vector<uint32_t> offsets;
    std::vector<TemporalEdge> edges;
    uint32_t vertices;
public:
    explicit StaticCSR(uint32_t v) : offsets(v + 1, 0), vertices(v) {}
    void rebuild(const std::vector<uint32_t>& src, const std::vector<uint32_t>& dst,
                 const std::vector<uint32_t>& ts, size_t count) {
        std::fill(offsets.begin(), offsets.end(), 0);
        for (size_t i = 0; i < count; ++i) ++offsets[src[i] + 1];
        for (uint32_t v = 0; v < vertices; ++v) offsets[v + 1] += offsets[v];
        edges.assign(count, {0, 0});
        std::vector<uint32_t> cursor(offsets.begin(), offsets.end() - 1);
        for (size_t i = 0; i < count; ++i) edges[cursor[src[i]]++] = {dst[i], ts[i]};
    }
    uint64_t scan_all() const {
        uint64_t sum = 0;
        for (const auto& e : edges) sum += e.target_node;
        return sum;
    }
    uint64_t scan_vertex(uint32_t v, uint32_t limit) const {
        const uint32_t begin = offsets[v], end = offsets[v + 1];
        const uint32_t take = std::min(limit, end - begin);
        uint64_t sum = 0;
        for (uint32_t i = end - take; i < end; ++i) sum += edges[i].target_node;
        return sum;
    }
};

struct Row {
    std::string layout;
    double build_seconds;
    double scan_seconds;
    double neighborhood_seconds;
};

void run_scale(uint32_t vertices, uint32_t num_edges, bool header) {
    std::mt19937 rng(12345);
    std::vector<uint32_t> src(num_edges), dst(num_edges), ts(num_edges);
    for (uint32_t i = 0; i < num_edges; ++i) {
        src[i] = rng() % vertices;
        dst[i] = rng() % vertices;
        ts[i] = 1000 + i;
    }

    // The access pattern the temporal sampler actually performs: jump to a
    // random vertex, read its most recent K edges.
    constexpr uint32_t FANOUT = 16;
    const uint32_t probes = std::min<uint32_t>(2'000'000, num_edges);
    std::vector<uint32_t> probe(probes);
    for (uint32_t i = 0; i < probes; ++i) probe[i] = rng() % vertices;

    std::vector<Row> rows;
    volatile uint64_t sink = 0;

    {   // PCSR
        PCSRGraph g(vertices, num_edges * 2, static_cast<size_t>(num_edges) * 64 + (1u << 26));
        auto t0 = Clock::now();
        for (uint32_t i = 0; i < num_edges; ++i) g.insert_edge(src[i], dst[i], ts[i]);
        const double build = seconds_since(t0);

        t0 = Clock::now();
        uint64_t sum = 0;
        for (uint32_t v = 0; v < vertices; ++v) {
            for (const TemporalEdge& edge : g.neighbors(v)) sum += edge.target_node;
        }
        const double scan = seconds_since(t0);
        sink += sum;

        t0 = Clock::now();
        sum = 0;
        for (uint32_t p = 0; p < probes; ++p) {
            const auto run = g.neighbors(probe[p]);
            const size_t take = std::min<size_t>(FANOUT, run.size());
            for (size_t i = run.size() - take; i < run.size(); ++i) sum += run[i].target_node;
        }
        const double nbr = seconds_since(t0);
        sink += sum;
        rows.push_back({"PCSR (packed, aligned)", build, scan, nbr});
    }

    {   // static CSR
        StaticCSR csr(vertices);
        auto t0 = Clock::now();
        csr.rebuild(src, dst, ts, num_edges);
        const double build = seconds_since(t0);
        t0 = Clock::now(); sink += csr.scan_all(); const double scan = seconds_since(t0);
        t0 = Clock::now();
        uint64_t sum = 0;
        for (uint32_t p = 0; p < probes; ++p) sum += csr.scan_vertex(probe[p], FANOUT);
        const double nbr = seconds_since(t0);
        sink += sum;
        rows.push_back({"static CSR (immutable)", build, scan, nbr});
    }

    {   // vector of vectors
        NaiveAdjacency naive(vertices);
        auto t0 = Clock::now();
        for (uint32_t i = 0; i < num_edges; ++i) naive.insert(src[i], dst[i], ts[i]);
        const double build = seconds_since(t0);
        t0 = Clock::now(); sink += naive.scan_all(); const double scan = seconds_since(t0);
        t0 = Clock::now();
        uint64_t sum = 0;
        for (uint32_t p = 0; p < probes; ++p) sum += naive.scan_vertex(probe[p], FANOUT);
        const double nbr = seconds_since(t0);
        sink += sum;
        rows.push_back({"vector<vector<Edge>>", build, scan, nbr});
    }

    {   // hash map of vectors
        HashAdjacency hash(vertices);
        auto t0 = Clock::now();
        for (uint32_t i = 0; i < num_edges; ++i) hash.insert(src[i], dst[i], ts[i]);
        const double build = seconds_since(t0);
        t0 = Clock::now(); sink += hash.scan_all(); const double scan = seconds_since(t0);
        t0 = Clock::now();
        uint64_t sum = 0;
        for (uint32_t p = 0; p < probes; ++p) sum += hash.scan_vertex(probe[p], FANOUT);
        const double nbr = seconds_since(t0);
        sink += sum;
        rows.push_back({"unordered_map<vector>", build, scan, nbr});
    }

    const double payload_mb = static_cast<double>(num_edges) * sizeof(TemporalEdge) / 1e6;
    if (header) {
        std::cout << "\n" << std::string(104, '=') << "\n"
                  << std::left << std::setw(24) << "layout"
                  << std::right << std::setw(12) << "build ns/e"
                  << std::setw(14) << "scan GB/s"
                  << std::setw(13) << "scan ns/e"
                  << std::setw(16) << "neighbourhood"
                  << std::setw(12) << "vs PCSR" << "\n"
                  << std::string(104, '-') << "\n";
    }
    std::cout << "-- " << num_edges << " edges over " << vertices
              << " vertices (" << std::fixed << std::setprecision(1) << payload_mb
              << " MB payload)\n";

    const double base_nbr = rows[0].neighborhood_seconds;
    for (const auto& r : rows) {
        const double gbps = payload_mb / 1000.0 / r.scan_seconds;
        std::cout << std::left << std::setw(24) << r.layout << std::right
                  << std::setw(12) << std::setprecision(1) << (r.build_seconds * 1e9 / num_edges)
                  << std::setw(14) << std::setprecision(2) << gbps
                  << std::setw(13) << std::setprecision(2) << (r.scan_seconds * 1e9 / num_edges)
                  << std::setw(13) << std::setprecision(1) << (r.neighborhood_seconds * 1e9 / probes) << " ns"
                  << std::setw(11) << std::setprecision(2) << (r.neighborhood_seconds / base_nbr) << "x"
                  << "\n";
    }
    if (sink == 0xFFFFFFFFFFFFFFFFull) std::cout << "";  // keep the sink alive
}

} // namespace

int main() {
    std::cout << "Layout comparison: does packing actually buy locality?\n"
              << "All four hold identical logical data. 'scan GB/s' is useful payload bytes\n"
              << "per second; 'neighbourhood' is the sampler's real pattern (random vertex,\n"
              << "16 most recent edges).\n";

    // Sweep the working set from comfortably inside cache out to well past it.
    const std::pair<uint32_t, uint32_t> scales[] = {
        {1'000, 100'000},        // ~0.8 MB  - fits L2
        {10'000, 1'000'000},     // ~8 MB    - around LLC
        {100'000, 10'000'000},   // ~80 MB   - DRAM
        {1'000'000, 40'000'000}, // ~320 MB  - firmly DRAM, sparse vertices
    };
    bool first = true;
    for (const auto& [vertices, edges] : scales) {
        run_scale(vertices, edges, first);
        first = false;
    }
    return 0;
}
