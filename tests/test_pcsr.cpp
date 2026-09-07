#include "pcsr_graph.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
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
    for (uint32_t v = 0; v < g.get_num_vertices(); ++v) {
        for (const TemporalEdge& e : g.neighbors(v)) {
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

    const uint32_t* starts = g.get_vertex_starts();

    uint64_t live = 0;
    uint64_t dead = 0;
    for (uint32_t v = 0; v < V; ++v) {
        check(offsets[v] <= offsets[v + 1], label + ": offsets are monotonic at vertex " +
                                                std::to_string(v));
        const uint32_t region = offsets[v + 1] - offsets[v];
        const uint32_t used = starts[v] + counts[v];
        check(used <= region, label + ": vertex " + std::to_string(v) +
                                  " does not overflow its region");

        // Live edges must occupy a packed run; gaps only ever follow them. The
        // expired prefix ahead of the run still holds its old edge records --
        // it is dead, not blanked -- so the gap check starts past it.
        for (uint32_t i = starts[v]; i < used; ++i) {
            check(edges[offsets[v] + i].target_node != EMPTY_GAP,
                  label + ": no hole inside vertex " + std::to_string(v) + "'s run");
        }
        for (uint32_t i = used; i < region; ++i) {
            check(edges[offsets[v] + i].target_node == EMPTY_GAP,
                  label + ": no live edge past vertex " + std::to_string(v) + "'s count");
        }
        live += counts[v];
        dead += starts[v];
    }
    check_eq(live, g.get_num_edges(), label + ": counts sum to reported edge total");
    check_eq(dead, g.get_dead_slots(), label + ": expired prefixes sum to dead_slots");
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

void test_differential_fuzz() {
    // Randomised operation sequences checked against a reference model.
    //
    // Every other test in this file is a scenario somebody thought of. The
    // failure modes this structure actually has -- a rebalance window that
    // straddles a resize, a hot vertex whose region migrates twice in a row --
    // live in interleavings nobody enumerates. So: random shapes, random
    // degrees, random capacities, and a std::multimap that must agree exactly,
    // including insertion order within each vertex and the relation on every
    // edge.
    std::cout << "Differential fuzz against a reference model\n";

    uint32_t configurations = 0;
    uint64_t total_edges = 0;
    uint32_t growth_seen = 0;
    uint32_t rebalance_seen = 0;
    bool all_match = true;

    for (uint32_t seed = 0; seed < 120; ++seed) {
        std::mt19937 rng(seed * 7919 + 13);

        const uint32_t vertices = 1 + rng() % 96;
        const uint32_t capacity = 1 + rng() % 512;
        const uint32_t operations = rng() % 6000;
        // Skew ranges from uniform to one vertex owning nearly everything.
        const uint32_t skew = rng() % 4;

        PCSRGraph graph(vertices, capacity, 48 * 1024 * 1024);
        std::map<uint32_t, std::vector<std::pair<uint32_t, EdgeRelation>>> oracle;

        uint32_t timestamp = 1;
        for (uint32_t op = 0; op < operations; ++op) {
            uint32_t src;
            switch (skew) {
                case 0:  src = rng() % vertices; break;                    // uniform
                case 1:  src = (rng() % 4 == 0) ? 0 : rng() % vertices; break;
                case 2:  src = rng() % std::max(1u, vertices / 8); break;  // narrow band
                default: src = 0; break;                                   // degenerate
            }
            const uint32_t dst = rng() % vertices;
            const EdgeRelation relation = static_cast<EdgeRelation>(rng() % 300 + 1);

            graph.insert_edge(src, dst, timestamp, relation);
            oracle[src].push_back({dst, relation});
            ++timestamp;
        }

        if (graph.get_resize_count() > 0) ++growth_seen;
        if (graph.get_rebalance_count() > 0) ++rebalance_seen;

        // Read the graph back and compare, vertex by vertex, in order.
        const uint32_t* offsets = graph.get_vertex_offsets();
        const uint32_t* counts = graph.get_vertex_counts();
        const TemporalEdge* edges = graph.get_edges();
        const EdgeRelation* relations = graph.get_edge_relations();

        bool matched = (graph.get_num_edges() == operations);
        for (uint32_t v = 0; v < vertices && matched; ++v) {
            const auto& expected = oracle[v];
            if (counts[v] != expected.size()) { matched = false; break; }
            for (uint32_t i = 0; i < counts[v]; ++i) {
                const uint32_t slot = offsets[v] + i;
                if (edges[slot].target_node != expected[i].first ||
                    relations[slot] != expected[i].second) {
                    matched = false;
                    break;
                }
            }
        }

        if (!matched) {
            all_match = false;
            std::cout << "    mismatch at seed " << seed << " (vertices=" << vertices
                      << " capacity=" << capacity << " ops=" << operations
                      << " skew=" << skew << ")\n";
        }
        ++configurations;
        total_edges += operations;
    }

    check(all_match, "every random configuration matches the reference model exactly");
    check(growth_seen > 10, "the fuzz corpus exercised PMA growth");
    check(rebalance_seen > 10, "the fuzz corpus exercised rebalancing");
    std::cout << "    " << configurations << " random configurations, "
              << total_edges << " edges, " << growth_seen << " with growth, "
              << rebalance_seen << " with rebalancing\n";
}

void test_arena_exhaustion_mid_resize() {
    // Growth allocates five arrays. If the arena runs dry on the third, the
    // graph must be left exactly as it was -- not half-migrated.
    //
    // It is safe by construction: resize_pma reseats no pointer until every
    // allocation has succeeded, so the failure is strongly exception-safe. That
    // is a property worth pinning down, because the obvious refactor -- assign
    // each pointer as it is allocated -- would quietly destroy it.
    std::cout << "Arena exhaustion during growth\n";

    for (size_t megabytes = 1; megabytes <= 4; ++megabytes) {
        PCSRGraph graph(64, 4096, megabytes * 1024 * 1024);
        uint64_t inserted = 0;
        bool threw = false;

        try {
            for (uint32_t k = 0; k < 2000000; ++k) {
                graph.insert_edge(k % 64, (k * 7) % 64, k + 1,
                                  static_cast<EdgeRelation>(k % 19 + 1));
                ++inserted;
            }
        } catch (const std::runtime_error&) {
            threw = true;
        }

        check(threw, "an undersized arena throws rather than overrunning");
        check_eq(graph.get_num_edges(), inserted,
                 "the failed insert did not increment the edge count");

        uint64_t scanned = 0;
        const uint32_t* offsets = graph.get_vertex_offsets();
        const uint32_t* counts = graph.get_vertex_counts();
        for (uint32_t v = 0; v < 64; ++v) {
            scanned += counts[v];
        }
        check_eq(scanned, inserted, "every edge inserted before the failure survives");
        check_eq(offsets[64], graph.get_edge_capacity(),
                 "the region map still covers the array after a failed resize");
        check_invariants(graph, "post-exhaustion");
    }
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

void test_edge_weights() {
    std::cout << "Edge weights ride alongside their edges\n";
    PCSRGraph g(64, 128, 16 * 1024 * 1024, /*store_weights=*/true);

    // Enough to force rebalances and at least one growth, so the weights have
    // to survive both copy paths, not just the O(1) insert.
    const uint32_t PER_VERTEX = 200;
    for (uint32_t v = 0; v < 64; ++v) {
        for (uint32_t i = 0; i < PER_VERTEX; ++i) {
            g.insert_edge(v, (v * 7 + i) % 64, 1000 + i,
                          static_cast<EdgeRelation>(i % 11),
                          static_cast<float>(v) + static_cast<float>(i) / 1024.0f);
        }
    }
    check(g.get_resize_count() > 0, "the case actually grew the PMA");
    check_invariants(g, "weights");

    bool intact = true;
    for (uint32_t v = 0; v < 64; ++v) {
        const auto run = g.neighbors(v);
        const auto weights = g.neighbor_weights(v);
        check_eq(run.size(), PER_VERTEX, "vertex keeps every edge");
        for (size_t i = 0; i < run.size(); ++i) {
            const float want = static_cast<float>(v) +
                               static_cast<float>(run[i].timestamp - 1000) / 1024.0f;
            if (weights[i] != want) intact = false;
        }
    }
    check(intact, "every weight still matches its edge after rebalance and growth");

    // The default must stay 0.0f so callers that never pass a weight are
    // unaffected.
    PCSRGraph plain(4, 16, 1024 * 1024, /*store_weights=*/true);
    plain.insert_edge(0, 1, 500);
    check(plain.neighbor_weights(0)[0] == 0.0f, "unspecified weight defaults to zero");

    // Weights are off by default, and a graph without them must refuse a
    // non-zero weight rather than drop it on the floor.
    PCSRGraph unweighted(4, 16, 1024 * 1024);
    check(!unweighted.has_weights(), "weights are off unless asked for");
    unweighted.insert_edge(0, 1, 500);
    check(unweighted.neighbor_weights(0).empty(), "an unweighted graph reports no weights");
    check_eq(unweighted.get_num_edges(), 1, "an unweighted graph still stores edges");

    bool threw = false;
    try { unweighted.insert_edge(0, 2, 501, 0, 1.5f); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a non-zero weight into an unweighted graph throws");
    check_eq(unweighted.get_num_edges(), 1, "the rejected insert did not land");
}

void test_expiry() {
    std::cout << "Expiring old edges\n";
    PCSRGraph g(32, 4096, 16 * 1024 * 1024, /*store_weights=*/true);

    for (uint32_t v = 0; v < 32; ++v) {
        for (uint32_t t = 0; t < 100; ++t) {
            g.insert_edge(v, (v + t) % 32, 1000 + t, 0, static_cast<float>(t));
        }
    }
    check_eq(g.get_num_edges(), 3200, "graph is fully populated");

    // Nothing older than the first timestamp, so this must be a no-op.
    check_eq(g.expire_before(1000), 0, "expiring before the oldest edge removes nothing");
    check_eq(g.get_dead_slots(), 0, "a no-op expiry leaves no dead slots");

    check_eq(g.expire_before(1040), 32 * 40, "expiry removes exactly the old prefix");
    check_eq(g.get_num_edges(), 3200 - 32 * 40, "edge count drops by what was removed");
    check_eq(g.get_expired_edges(), 32 * 40, "lifetime expiry counter tracks removals");
    check_eq(g.get_dead_slots(), 32 * 40, "dead slots are held until a rebalance");
    check_invariants(g, "after expiry");

    // Survivors must be exactly the recent ones, still in order, still paired
    // with their own weights.
    bool survivors_correct = true;
    for (uint32_t v = 0; v < 32; ++v) {
        const auto run = g.neighbors(v);
        const auto weights = g.neighbor_weights(v);
        if (run.size() != 60) { survivors_correct = false; continue; }
        for (size_t i = 0; i < run.size(); ++i) {
            if (run[i].timestamp != 1040 + i) survivors_correct = false;
            if (weights[i] != static_cast<float>(40 + i)) survivors_correct = false;
        }
    }
    check(survivors_correct, "survivors are the recent edges, ordered, with their weights");

    // Inserting after an expiry must land past the dead prefix, not on top of
    // a survivor.
    g.insert_edge(0, 5, 9999, 0, 42.0f);
    const auto run = g.neighbors(0);
    check_eq(run.size(), 61, "insert after expiry appends");
    check_eq(run[60].timestamp, 9999, "the appended edge is the newest");
    check_eq(run[0].timestamp, 1040, "the oldest survivor is untouched");
    check_invariants(g, "insert after expiry");

    // Expiring everything is legal and leaves an empty but valid graph.
    PCSRGraph all(8, 256, 4 * 1024 * 1024);
    for (uint32_t i = 0; i < 100; ++i) all.insert_edge(i % 8, (i + 1) % 8, 1000 + i);
    check_eq(all.expire_before(2000), 100, "expiring past the newest empties the graph");
    check_eq(all.get_num_edges(), 0, "no edges remain");
    check_invariants(all, "fully expired");
    all.insert_edge(3, 4, 3000);
    check_eq(all.get_num_edges(), 1, "an emptied graph still accepts inserts");
    check_invariants(all, "refilled after full expiry");
}

void test_expiry_reclaims_space() {
    std::cout << "Expired slots are reclaimed by rebalancing\n";
    // A sliding window: insert a batch, expire the batch before it, repeat.
    // Without reclamation the PMA would grow without bound; the point of the
    // test is that it does not.
    PCSRGraph g(16, 2048, 8 * 1024 * 1024);

    const uint32_t BATCHES = 400;
    const uint32_t PER_BATCH = 16;
    uint32_t now = 1000;
    for (uint32_t b = 0; b < BATCHES; ++b) {
        for (uint32_t i = 0; i < PER_BATCH; ++i) {
            g.insert_edge(i % 16, (i * 3) % 16, now);
        }
        ++now;
        // Keep only the last two batches.
        if (b >= 2) g.expire_before(now - 2);
    }

    check(g.get_num_edges() <= 2 * PER_BATCH,
          "the window holds only the recent batches");
    check_invariants(g, "sliding window");

    // 400 batches at 16 edges each is 6,400 insertions against a 2,048-slot
    // PMA. If expired space were never returned the array would have had to
    // double at least twice; reclamation is what keeps it at its original size.
    check_eq(g.get_resize_count(), 0, "a bounded window never grows the PMA");
    check(g.get_dead_slots() < g.get_edge_capacity(),
          "dead slots do not accumulate to fill the array");
    check_eq(g.get_expired_edges(), BATCHES * PER_BATCH - g.get_num_edges(),
             "every edge that left the window is accounted for");
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
    test_differential_fuzz();
    test_arena_exhaustion_mid_resize();
    test_boundaries();
    test_cache_alignment();
    test_edge_weights();
    test_expiry();
    test_expiry_reclaims_space();
    test_arena_exhaustion();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    if (g_failures > 0) {
        std::cout << g_failures << " FAILURES\n";
        return 1;
    }
    std::cout << "All assertions passed.\n";
    return 0;
}
