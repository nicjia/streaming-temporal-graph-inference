#include "spsc_queue.hpp"
#include "streaming_ingestor.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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

void test_sizing() {
    std::cout << "Capacity and sizing\n";

    // Capacity 1 used to construct fine and then reject every push forever,
    // which is indistinguishable from a permanently full queue.
    bool threw = false;
    try { SPSCQueue<int> q(1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "capacity below 2 is rejected rather than silently unusable");

    SPSCQueue<int> q(5);
    check_eq(q.capacity(), 8, "capacity rounds up to a power of two");
    check_eq(q.usable_capacity(), 7, "one slot is reserved for full/empty disambiguation");
    check(q.empty(), "a fresh queue is empty");

    for (int i = 0; i < 7; ++i) {
        check(q.push(i), "push " + std::to_string(i) + " into a queue with room");
    }
    check(!q.push(99), "push fails once every usable slot is taken");
    check_eq(q.size(), 7, "size reports the usable capacity when full");

    int value = -1;
    check(q.pop(value) && value == 0, "pop returns items in FIFO order");
    check(q.push(99), "a slot freed by pop becomes available again");
}

void test_empty_pop() {
    std::cout << "Empty and drained behaviour\n";
    SPSCQueue<int> q(4);
    int value = 123;
    check(!q.pop(value), "pop on an empty queue returns false");
    check_eq(value, 123, "a failed pop leaves the output untouched");

    q.push(7);
    check(q.pop(value) && value == 7, "pop after push returns the item");
    check(!q.pop(value), "queue is empty again after draining");
}

void test_wraparound() {
    std::cout << "Index wraparound\n";
    // Push and pop far more items than the buffer holds, so head and tail wrap
    // many times. An off-by-one in the mask shows up here and nowhere else.
    SPSCQueue<uint64_t> q(8);
    uint64_t out = 0;
    bool ordered = true;

    for (uint64_t i = 0; i < 10000; ++i) {
        check_eq(q.push(i), 1, "");
        --g_checks;  // do not count 10k identical assertions
        if (!q.pop(out) || out != i) {
            ordered = false;
        }
    }
    check(ordered, "10,000 push/pop cycles wrap correctly and stay in order");
    check(q.empty(), "queue is empty after equal pushes and pops");
}

void test_concurrent_handoff() {
    std::cout << "Concurrent producer and consumer\n";

    // The property that matters: every item the producer sends is received
    // exactly once, in order, with no duplicates and none lost. A lock-free
    // queue can pass every single-threaded test and still fail this.
    constexpr uint64_t COUNT = 2'000'000;
    SPSCQueue<uint64_t> q(1024);

    std::atomic<bool> order_violated{false};
    std::atomic<uint64_t> received{0};

    const auto start = std::chrono::steady_clock::now();

    std::thread consumer([&] {
        uint64_t expected = 0;
        uint64_t value = 0;
        while (expected < COUNT) {
            if (q.pop(value)) {
                if (value != expected) {
                    order_violated.store(true);
                }
                ++expected;
            }
        }
        received.store(expected);
    });

    for (uint64_t i = 0; i < COUNT; ++i) {
        while (!q.push(i)) {
            // Spin: a full queue is back-pressure, not an error.
        }
    }
    consumer.join();

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    check_eq(received.load(), COUNT, "every item crosses the queue exactly once");
    check(!order_violated.load(), "items arrive in the order they were sent");
    check(q.empty(), "queue is drained at the end of the handoff");

    std::cout << "  " << static_cast<uint64_t>(COUNT / seconds / 1e6)
              << "M items/sec across the thread boundary\n";
}

void test_backpressure() {
    std::cout << "Back-pressure under a slow consumer\n";

    // A tiny queue forces the producer to block constantly. This is the
    // scheduling pattern most likely to expose a missing memory barrier.
    constexpr uint64_t COUNT = 200'000;
    SPSCQueue<uint64_t> q(2);
    std::atomic<uint64_t> sum{0};

    std::thread consumer([&] {
        uint64_t value = 0;
        uint64_t total = 0;
        for (uint64_t i = 0; i < COUNT; ++i) {
            while (!q.pop(value)) {
            }
            total += value;
        }
        sum.store(total);
    });

    for (uint64_t i = 0; i < COUNT; ++i) {
        while (!q.push(i)) {
        }
    }
    consumer.join();

    const uint64_t expected = COUNT * (COUNT - 1) / 2;
    check_eq(sum.load(), expected, "no item is lost or duplicated at capacity 1");
}

void test_struct_payload() {
    std::cout << "Non-trivial payload\n";
    struct Event {
        uint32_t src;
        uint32_t dst;
        uint32_t timestamp;
        uint16_t relation;
    };

    SPSCQueue<Event> q(16);
    q.push({1, 2, 300, 7});
    Event out{};
    check(q.pop(out), "struct payload pops");
    check(out.src == 1 && out.dst == 2 && out.timestamp == 300 && out.relation == 7,
          "struct fields survive the round trip");
}

void test_streaming_ingestor() {
    std::cout << "Streaming ingestion through the queue\n";

    constexpr uint32_t VERTICES = 512;
    constexpr uint32_t EVENTS = 300000;
    PCSRGraph graph(VERTICES, EVENTS * 2, 128u * 1024 * 1024);

    {
        // Deliberately small queue so the producer hits back-pressure
        // constantly; that is the interleaving most likely to lose an event.
        StreamingIngestor ingestor(graph, 256);
        ingestor.start();
        check(ingestor.is_running(), "ingestor reports running after start");

        for (uint32_t i = 0; i < EVENTS; ++i) {
            ingestor.push({i % VERTICES, (i * 7) % VERTICES, 1000 + i,
                           static_cast<EdgeRelation>(i % 19 + 1)});
        }
        ingestor.drain();
        check_eq(ingestor.get_consumed(), EVENTS, "consumer drains everything pushed");
        check(ingestor.get_producer_spins() > 0,
              "a small queue actually exercised back-pressure");
        ingestor.stop();
        check(!ingestor.is_running(), "ingestor reports stopped");
    }

    check_eq(graph.get_num_edges(), EVENTS, "every streamed event reached the graph");

    uint64_t scanned = 0;
    bool relations_intact = true;
    for (uint32_t v = 0; v < VERTICES; ++v) {
        const auto run = graph.neighbors(v);
        const auto rels = graph.neighbor_relations(v);
        scanned += run.size();
        for (size_t i = 0; i < run.size(); ++i) {
            // relation was derived from the event index, which also set the
            // timestamp, so the pairing is checkable after the fact.
            const uint32_t index = run[i].timestamp - 1000;
            if (rels[i] != static_cast<EdgeRelation>(index % 19 + 1)) relations_intact = false;
        }
    }
    check_eq(scanned, EVENTS, "a full scan agrees with the reported count");
    check(relations_intact, "relations survive the queue handoff intact");

    // The guard must be released once streaming stops.
    graph.insert_edge(0, 1, 999999);
    check_eq(graph.get_num_edges(), EVENTS + 1, "direct writes work again after stop()");
}

void test_streaming_exclusivity() {
    std::cout << "Streaming holds the write guard\n";
    PCSRGraph graph(16, 256, 4u * 1024 * 1024);

    StreamingIngestor first(graph, 64);
    first.start();

    // start() must not return until the guard is actually held, otherwise a
    // caller racing immediately afterwards slips in ahead of it.
    bool rejected = false;
    try {
        PCSRGraph::WriteGuard guard(graph);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "an active stream blocks other writers immediately after start()");

    StreamingIngestor second(graph, 64);
    bool second_rejected = false;
    try {
        second.start();
    } catch (const std::runtime_error&) {
        second_rejected = true;
    }
    check(second_rejected, "a second ingestor on the same graph is refused");

    first.stop();
    bool now_allowed = true;
    try {
        PCSRGraph::WriteGuard guard(graph);
    } catch (const std::runtime_error&) {
        now_allowed = false;
    }
    check(now_allowed, "the guard is released when streaming stops");
}

} // namespace

int main() {
    std::cout << "Running SPSCQueue Tests\n\n";

    test_sizing();
    test_empty_pop();
    test_wraparound();
    test_concurrent_handoff();
    test_backpressure();
    test_struct_payload();
    test_streaming_ingestor();
    test_streaming_exclusivity();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
    if (g_failures > 0) {
        std::cout << g_failures << " FAILURES\n";
        return 1;
    }
    std::cout << "All assertions passed.\n";
    return 0;
}
