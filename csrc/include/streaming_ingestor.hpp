#pragma once
#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>

#include "pcsr_graph.hpp"
#include "spsc_queue.hpp"
#include "types.hpp"

/**
 * @brief Streams events into a PCSRGraph across a lock-free queue.
 *
 * This is the path the SPSC queue was written for. A producer -- typically a
 * Python thread polling a feed -- pushes events; a dedicated C++ consumer
 * thread drains them into the graph. The two run concurrently and never share
 * a lock.
 *
 * Why bother, when insert_edges() already batches? Because batching requires
 * the whole batch to exist before any of it is inserted. A live feed arrives
 * continuously, and the useful property is that decoding the next chunk in
 * Python overlaps with inserting the previous one in C++, rather than
 * alternating. The queue is the handoff that makes that overlap possible.
 *
 * The consumer thread holds the graph's WriteGuard for its entire lifetime, so
 * the single-writer invariant is enforced structurally: any other thread
 * attempting to insert while streaming is active is rejected rather than
 * silently interleaving.
 */
class StreamingIngestor {
public:
    struct Event {
        uint32_t src;
        uint32_t dst;
        uint32_t timestamp;
        EdgeRelation relation;
        float weight;
    };

private:
    PCSRGraph& graph;
    SPSCQueue<Event> queue;
    std::thread consumer;

    std::atomic<bool> running;
    std::atomic<bool> stopping;
    std::atomic<uint64_t> pushed;
    std::atomic<uint64_t> consumed;
    std::atomic<uint64_t> spins;      // producer waits: how often the queue filled
    std::atomic<uint64_t> idle_polls; // consumer waits: how often it outran the producer

    // Set if the consumer dies on a bad event so the producer can surface it
    // rather than blocking forever against a queue nobody is draining.
    std::atomic<bool> failed;
    std::string failure_message;

    void consume(std::promise<void> ready) {
        // The guard is taken here rather than in start() because it must be
        // held by the thread that actually mutates. But start() has to know it
        // succeeded before returning: otherwise it races the consumer, and a
        // caller who writes to the graph immediately after start() slips in
        // ahead of the guard. The promise closes that window and carries the
        // rejection back to start() if the graph already has a writer.
        std::unique_ptr<PCSRGraph::WriteGuard> guard;
        try {
            guard = std::make_unique<PCSRGraph::WriteGuard>(graph);
        } catch (...) {
            ready.set_exception(std::current_exception());
            return;
        }
        ready.set_value();

        Event event{};
        try {
            while (true) {
                if (queue.pop(event)) {
                    graph.insert_edge(event.src, event.dst, event.timestamp, event.relation,
                                      event.weight);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                } else if (stopping.load(std::memory_order_acquire)) {
                    // One last drain: stop() may have been signalled while
                    // items were still in flight.
                    if (!queue.pop(event)) break;
                    graph.insert_edge(event.src, event.dst, event.timestamp, event.relation,
                                      event.weight);
                    consumed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    idle_polls.fetch_add(1, std::memory_order_relaxed);
                    std::this_thread::yield();
                }
            }
        } catch (const std::exception& error) {
            failure_message = error.what();
            failed.store(true, std::memory_order_release);
        }
    }

public:
    StreamingIngestor(PCSRGraph& target, size_t queue_capacity = 65536)
        : graph(target), queue(queue_capacity), running(false), stopping(false),
          pushed(0), consumed(0), spins(0), idle_polls(0), failed(false) {}

    ~StreamingIngestor() {
        if (running.load(std::memory_order_acquire)) {
            try { stop(); } catch (...) {}
        }
    }

    StreamingIngestor(const StreamingIngestor&) = delete;
    StreamingIngestor& operator=(const StreamingIngestor&) = delete;

    void start() {
        if (running.exchange(true, std::memory_order_acq_rel)) {
            throw std::runtime_error("StreamingIngestor is already running");
        }
        stopping.store(false, std::memory_order_release);
        failed.store(false, std::memory_order_release);

        std::promise<void> ready;
        std::future<void> acquired = ready.get_future();
        consumer = std::thread([this, promise = std::move(ready)]() mutable {
            consume(std::move(promise));
        });

        try {
            acquired.get();  // rethrows if the guard could not be taken
        } catch (...) {
            if (consumer.joinable()) consumer.join();
            running.store(false, std::memory_order_release);
            throw;
        }
    }

    /**
     * Producer side. Blocks (spinning) while the queue is full, which is
     * back-pressure rather than an error -- dropping events would silently
     * corrupt the history the temporal model depends on.
     */
    void push(const Event& event) {
        while (!queue.push(event)) {
            if (failed.load(std::memory_order_acquire)) {
                throw std::runtime_error("streaming consumer failed: " + failure_message);
            }
            if (!running.load(std::memory_order_acquire)) {
                throw std::runtime_error("StreamingIngestor is not running");
            }
            spins.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
        pushed.fetch_add(1, std::memory_order_relaxed);
    }

    /// Signals the consumer to finish the queue and joins it.
    void stop() {
        if (!running.load(std::memory_order_acquire)) return;
        stopping.store(true, std::memory_order_release);
        if (consumer.joinable()) consumer.join();
        running.store(false, std::memory_order_release);
        if (failed.load(std::memory_order_acquire)) {
            throw std::runtime_error("streaming consumer failed: " + failure_message);
        }
    }

    /// Blocks until the consumer has caught up with everything pushed so far.
    void drain() {
        while (consumed.load(std::memory_order_acquire) <
               pushed.load(std::memory_order_acquire)) {
            if (failed.load(std::memory_order_acquire)) {
                throw std::runtime_error("streaming consumer failed: " + failure_message);
            }
            std::this_thread::yield();
        }
    }

    bool is_running() const { return running.load(std::memory_order_acquire); }
    uint64_t get_pushed() const { return pushed.load(std::memory_order_acquire); }
    uint64_t get_consumed() const { return consumed.load(std::memory_order_acquire); }
    uint64_t get_producer_spins() const { return spins.load(std::memory_order_acquire); }
    uint64_t get_consumer_idle_polls() const { return idle_polls.load(std::memory_order_acquire); }
    size_t get_queue_capacity() const { return queue.usable_capacity(); }
};
