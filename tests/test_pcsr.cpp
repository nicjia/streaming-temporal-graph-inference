#include "pcsr_graph.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cout << "  FAIL: " << what << "\n";
    }
}

void check_eq(uint64_t got, uint64_t want, const std::string& what) {
    ++g_checks;
    if (got != want) {
        ++g_failures;
        std::cout << "  FAIL: " << what << " (got " << got << ", want " << want << ")\n";
    }
}

using EdgeList = std::multimap<uint32_t, std::pair<uint32_t, uint32_t>>;

/// Reads every live edge back out of the PMA, exactly as a consumer would.
EdgeList scan_graph(const PCSRGraph& g) {
    EdgeList found;
    const uint32_t* offsets = g.get_vertex_offsets();
    const uint32_t* counts = g.get_vertex_counts();
    const TemporalEdge* edges = g.get_edges();

    for (uint32_t v = 0; v < g.get_num_vertices(); ++v) {
        for (uint32_t i = 0; i < counts[v]; ++i) {
            const TemporalEdge& e = edges[offsets[v] + i];
            found.insert({v, {e.target_node, e.timestamp}});
        }
    }
    return found;
}

/// The structural contract every operation must leave intact.
void check_invariants(const PCSRGraph& g, const std::string& label) {
    const uint32_t* offsets = g.get_vertex_offsets();
    const uint32_t* counts = g.get_vertex_counts();
    const TemporalEdge* edges = g.get_edges();
    const uint32_t V = g.get_num_vertices();

    check_eq(offsets[0], 0, label + ": first region starts at 0");
    check_eq(offsets[V], g.get_edge_capacity(), label + ": last region ends at capacity");

    uint64_t live = 0;
    for (uint32_t v = 0; v < V; ++v) {
        check(offsets[v] <= offsets[v + 1], label + ": offsets are monotonic at vertex " +
                                                std::to_string(v));
        const uint32_t region = offsets[v + 1] - offsets[v];
        check(counts[v] <= region, label + ": vertex " + std::to_string(v) +
                                       " does not overflow its region");

        // Live edges must occupy a packed prefix; gaps only ever follow them.
        for (uint32_t i = 0; i < counts[v]; ++i) {
            check(edges[offsets[v] + i].target_node != EMPTY_GAP,
                  label + ": no hole inside vertex " + std::to_string(v) + "'s run");
        }
        for (uint32_t i = counts[v]; i < region; ++i) {
            check(edges[offsets[v] + i].target_node == EMPTY_GAP,
                  label + ": no live edge past vertex " + std::to_string(v) + "'s count");
        }
        live += counts[v];
    }
    check_eq(live, g.get_num_edges(), label + ": counts sum to reported edge total");
}

/// Insert `edges` and assert not a single one is lost or altered.
void run_retention_case(const std::string& label, uint32_t vertices, uint32_t capacity,
                        const std::vector<std::vector<uint32_t>>& adjacency) {
    std::cout << label << "\n";
    PCSRGraph g(vertices, capacity, 64 * 1024 * 1024);

    EdgeList expected;
    uint32_t ts = 1000;
    for (uint32_t src = 0; src < adjacency.size(); ++src) {
        for (uint32_t dst : adjacency[src]) {
            g.insert_edge(src, dst, ts);
            expected.insert({src, {dst, ts}});
            ++ts;
        }
    }

    const EdgeList found = scan_graph(g);
    check_eq(g.get_num_edges(), expected.size(), label + ": reported edge count");
    check_eq(found.size(), expected.size(), label + ": edges retained in the PMA");
    check(found == expected, label + ": every (src, dst, timestamp) survives intact");

    for (uint32_t v = 0; v < vertices; ++v) {
        check_eq(g.get_degree(v), adjacency.size() > v ? adjacency[v].size() : 0,
                 label + ": degree of vertex " + std::to_string(v));
    }
    check_invariants(g, label);
}

void test_basic_layout() {
    std::cout << "Basic layout and ordering\n";
    PCSRGraph graph(10, 100, 10 * 1024 * 1024);

    graph.insert_edge(0, 5, 1000);
    graph.insert_edge(0, 6, 1001);
    graph.insert_edge(1, 2, 1002);

    const uint32_t* offsets = graph.get_vertex_offsets();
    const TemporalEdge* edges = graph.get_edges();
    const uint32_t start_0 = offsets[0];

    // Insertion order is preserved within a vertex's run.
    check_eq(edges[start_0].target_node, 5, "node 0 first edge target");
    check_eq(edges[start_0].timestamp, 1000, "node 0 first edge timestamp");
    check_eq(edges[start_0 + 1].target_node, 6, "node 0 second edge target");
    check_eq(edges[start_0 + 1].timestamp, 1001, "node 0 second edge timestamp");
    check_eq(edges[offsets[1]].target_node, 2, "node 1 first edge target");

    check_eq(graph.get_num_edges(), 3, "total edge count");
    check_invariants(graph, "basic");
}

void test_uniform_fill() {
    // 10 vertices x 12 edges into 120 slots: exercises the fast path plus a few
    // rebalances as regions tighten.
    std::vector<std::vector<uint32_t>> adjacency(10);
    for (uint32_t v = 0; v < 10; ++v) {
        for (uint32_t k = 0; k < 12; ++k) {
            adjacency[v].push_back((v + k) % 10);
        }
    }
    run_retention_case("Uniform degree, 120 edges into 120 slots", 10, 120, adjacency);
}

void test_single_hot_vertex() {
    // The case that silently dropped 29 of 40 edges before: one vertex whose
    // degree is 4x its initial region, with idle neighbours to borrow from.
    std::vector<std::vector<uint32_t>> adjacency(10);
    for (uint32_t k = 0; k < 40; ++k) {
        adjacency[3].push_back(k % 10);
    }
    run_retention_case("Single hot vertex, degree 40 in a 10-slot region", 10, 100, adjacency);
}

void test_power_law_skew() {
    // GDELT-shaped: a handful of actors carry most of the events.
    std::mt19937 rng(1234);
    const uint32_t V = 256;
    std::vector<std::vector<uint32_t>> adjacency(V);

    for (uint32_t v = 0; v < V; ++v) {
        const uint32_t degree = (v < 4) ? 400 : (v < 32 ? 20 : 1);
        for (uint32_t k = 0; k < degree; ++k) {
            adjacency[v].push_back(rng() % V);
        }
    }
    run_retention_case("Power-law skew, 4 vertices holding most edges", V, 4096, adjacency);
}

void test_growth_beyond_capacity() {
    // Deliberately undersized: forces several resize_pma() doublings.
    std::mt19937 rng(99);
    const uint32_t V = 64;
    std::vector<std::vector<uint32_t>> adjacency(V);
    for (uint32_t k = 0; k < 5000; ++k) {
        adjacency[rng() % V].push_back(rng() % V);
    }
    run_retention_case("Growth: 5000 edges into an initial 64 slots", V, 64, adjacency);
}

void test_all_edges_on_one_vertex() {
    // Pathological: the entire graph hangs off vertex 0, so every rebalance
    // escalates to the full array and then to a resize.
    std::vector<std::vector<uint32_t>> adjacency(8);
    for (uint32_t k = 0; k < 1000; ++k) {
        adjacency[0].push_back(k % 8);
    }
    run_retention_case("Degenerate: all 1000 edges on vertex 0", 8, 16, adjacency);
}

void test_chronological_order_preserved() {
    // The Python temporal sampler binary-searches each vertex's run for the
    // events preceding a query time. That is only valid if a run stays sorted
    // by timestamp, which holds iff rebalancing and growth preserve insertion
    // order. Nothing else in the suite pins that down directly.
    std::cout << "Chronological order survives rebalance and growth\n";

    std::mt19937 rng(2024);
    const uint32_t V = 128;
    PCSRGraph g(V, 128, 32 * 1024 * 1024); // deliberately tiny: forces growth

    // Insert in strictly increasing timestamp order, as a replay would.
    uint32_t ts = 1;
    for (uint32_t k = 0; k < 20000; ++k) {
        const uint32_t src = (k % 3 == 0) ? 0 : (rng() % V); // one hot vertex
        g.insert_edge(src, rng() % V, ts++);
    }

    check(g.get_resize_count() > 0, "the test actually exercised growth");
    check(g.get_rebalance_count() > 0, "the test actually exercised rebalancing");

    const uint32_t* offsets = g.get_vertex_offsets();
    const uint32_t* counts = g.get_vertex_counts();
    const TemporalEdge* edges = g.get_edges();

    bool sorted_everywhere = true;
    for (uint32_t v = 0; v < V; ++v) {
        for (uint32_t i = 1; i < counts[v]; ++i) {
            if (edges[offsets[v] + i - 1].timestamp > edges[offsets[v] + i].timestamp) {
                sorted_everywhere = false;
            }
        }
    }
    check(sorted_everywhere, "every adjacency run is still ascending in time");
    check_eq(g.get_num_edges(), 20000, "no edges lost while growing");
    check_invariants(g, "chronological");
}

void test_edge_relations() {
    // Relations live in an array parallel to the edges. Parallel arrays are
    // exactly the kind of thing that drifts out of sync during a rebalance, and
    // when they do nothing crashes -- edges just silently acquire the wrong
    // relation type. So this checks the pairing survives both rebalancing and
    // growth, not merely that relations can be stored.
    std::cout << "Edge relations survive rebalance and growth\n";

    std::mt19937 rng(4242);
    const uint32_t V = 64;
    PCSRGraph g(V, 64, 32 * 1024 * 1024); // tiny: forces growth

    // The relation is derived from the edge itself, so any mismatch after a
    // rebalance is detectable without keeping a side table.
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, EdgeRelation>> inserted;
    uint32_t ts = 1;
    for (uint32_t k = 0; k < 12000; ++k) {
        const uint32_t src = (k % 4 == 0) ? 0 : (rng() % V); // one hot vertex
        const uint32_t dst = rng() % V;
        const EdgeRelation relation = static_cast<EdgeRelation>((dst * 7 + 3) % 20 + 1);
        g.insert_edge(src, dst, ts, relation);
        inserted.push_back({src, dst, ts, relation});
        ++ts;
    }

    check(g.get_resize_count() > 0, "relations test exercised growth");
    check(g.get_rebalance_count() > 0, "relations test exercised rebalancing");

    const uint32_t* offsets = g.get_vertex_offsets();
    const uint32_t* counts = g.get_vertex_counts();
    const TemporalEdge* edges = g.get_edges();
    const EdgeRelation* relations = g.get_edge_relations();

    uint64_t seen = 0;
    bool paired = true;
    bool nonzero = false;
    for (uint32_t v = 0; v < V; ++v) {
        for (uint32_t i = 0; i < counts[v]; ++i) {
            const uint32_t slot = offsets[v] + i;
            const EdgeRelation expected =
                static_cast<EdgeRelation>((edges[slot].target_node * 7 + 3) % 20 + 1);
            if (relations[slot] != expected) {
                paired = false;
            }
            if (relations[slot] != RELATION_UNKNOWN) {
                nonzero = true;
            }
            ++seen;
        }
    }
    check(nonzero, "relations are actually populated");
    check(paired, "every edge still carries its own relation after rebalancing");
    check_eq(seen, inserted.size(), "no edge lost while carrying relations");
    check_invariants(g, "relations");

    // The default keeps existing callers working unchanged.
    PCSRGraph plain(4, 32, 1024 * 1024);
    plain.insert_edge(0, 1, 100);
    check_eq(plain.get_edge_relations()[plain.get_vertex_offsets()[0]], RELATION_UNKNOWN,
             "omitting the relation defaults to RELATION_UNKNOWN");
}

void test_boundaries() {
    std::cout << "Boundary conditions\n";

    // A single vertex has no neighbours to rebalance against; growth is the
    // only escape hatch.
    {
        PCSRGraph g(1, 1, 4 * 1024 * 1024);
        for (uint32_t k = 0; k < 100; ++k) {
            g.insert_edge(0, 0, 500 + k);
        }
        check_eq(g.get_num_edges(), 100, "single-vertex graph retains all self-loops");
        check_eq(scan_graph(g).size(), 100, "single-vertex graph scan");
        check_invariants(g, "single vertex");
    }

    // capacity < vertices used to leave every region zero-width.
    {
        PCSRGraph g(100, 50, 4 * 1024 * 1024);
        check(g.get_edge_capacity() >= 100, "capacity is clamped up to the vertex count");
        check_eq(g.get_vertex_offsets()[100], g.get_edge_capacity(), "regions cover the array");
        for (uint32_t v = 0; v < 100; ++v) {
            check(g.get_vertex_offsets()[v + 1] > g.get_vertex_offsets()[v],
                  "vertex " + std::to_string(v) + " has a non-empty region");
        }
        g.insert_edge(99, 0, 7);
        check_eq(g.get_num_edges(), 1, "insert into clamped graph");
    }

    // Out-of-range endpoints must throw, not corrupt memory.
    {
        PCSRGraph g(4, 16, 1024 * 1024);
        bool threw = false;
        try { g.insert_edge(4, 0, 1); } catch (const std::out_of_range&) { threw = true; }
        check(threw, "src >= num_vertices throws");
        threw = false;
        try { g.insert_edge(0, 4, 1); } catch (const std::out_of_range&) { threw = true; }
        check(threw, "dst >= num_vertices throws");
        check_eq(g.get_num_edges(), 0, "rejected inserts do not change the graph");
    }

    // Zero vertices is meaningless and used to divide by zero.
    {
        bool threw = false;
        try { PCSRGraph g(0, 16, 1024 * 1024); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "zero-vertex graph throws");
    }
}

void test_cache_alignment() {
    std::cout << "Cache-line alignment\n";
    PCSRGraph g(1000, 10000, 8 * 1024 * 1024);

    check_eq(reinterpret_cast<uintptr_t>(g.get_vertex_offsets()) % 64, 0,
             "vertex_offsets is 64-byte aligned");
    check_eq(reinterpret_cast<uintptr_t>(g.get_vertex_counts()) % 64, 0,
             "vertex_counts is 64-byte aligned");
    check_eq(reinterpret_cast<uintptr_t>(g.get_edges()) % 64, 0,
             "edges is 64-byte aligned");
}

void test_arena_exhaustion() {
    std::cout << "Arena exhaustion\n";
    // 1 MB cannot back a 10M-slot PMA; this must be a clean throw, not a crash.
    bool threw = false;
    try { PCSRGraph g(1000, 10'000'000, 1024 * 1024); }
    catch (const std::runtime_error&) { threw = true; }
    check(threw, "over-subscribed arena throws instead of overrunning");
}

} // namespace

int main() {
    std::cout << "Running PCSRGraph Unit Tests\n\n";

    test_basic_layout();
    test_uniform_fill();
    test_single_hot_vertex();
    test_power_law_skew();
    test_growth_beyond_capacity();
    test_all_edges_on_one_vertex();
    test_chronological_order_preserved();
    test_edge_relations();
    test_boundaries();
    test_cache_alignment();
    test_arena_exhaustion();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    if (g_failures > 0) {
        std::cout << g_failures << " FAILURES\n";
        return 1;
    }
    std::cout << "All assertions passed.\n";
    return 0;
}
