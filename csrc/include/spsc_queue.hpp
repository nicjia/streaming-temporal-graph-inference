#pragma once
#include <atomic>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @brief Wait-free single-producer / single-consumer ring buffer.
 *
 * Exactly one thread may call push() and exactly one other may call pop().
 * With more than one of either the invariants below do not hold.
 *
 * Capacity is rounded up to a power of two so the wrap is a bitmask rather
 * than a modulo. That is not micro-optimisation for its own sake: at the rates
 * this queue exists to sustain, an integer division on every push and every
 * pop is a substantial fraction of the total work, and the rounding costs at
 * most a factor of two in memory on a buffer that is allocated once.
 *
 * One slot is always left empty, because a ring buffer whose head and tail are
 * equal cannot otherwise distinguish full from empty. So a queue built with
 * capacity N holds N-1 items. This used to be silent, and a queue of capacity 1
 * was permanently unusable -- push simply returned false forever, which looks
 * identical to a full queue.
 */
template <typename T>
class SPSCQueue {
    std::vector<T> buffer;
    size_t mask;

    // head and tail sit on separate cache lines. Without the padding the
    // producer's store to tail would invalidate the consumer's cached copy of
    // head on every single push, turning a wait-free queue into a cache-line
    // ping-pong.
    alignas(64) std::atomic<size_t> head;
    alignas(64) std::atomic<size_t> tail;

    // Each side caches the other's index and only re-reads the atomic when its
    // cached view says there is no room (or nothing to take). In the common
    // case where the queue is neither full nor empty this removes one acquire
    // load from every operation.
    alignas(64) size_t cached_head;
    alignas(64) size_t cached_tail;

    static size_t round_up_pow2(size_t value) {
        size_t result = 1;
        while (result < value) {
            result <<= 1;
        }
        return result;
    }

public:
    /**
     * @param requested Minimum usable slots + 1. Rounded up to a power of two;
     *        the usable count is then capacity() - 1. Must be at least 2.
     */
    explicit SPSCQueue(size_t requested)
        : buffer(round_up_pow2(requested < 2 ? 2 : requested)),
          mask(buffer.size() - 1),
          head(0), tail(0), cached_head(0), cached_tail(0) {
        if (requested < 2) {
            throw std::invalid_argument(
                "SPSCQueue needs capacity >= 2: one slot is reserved to "
                "distinguish full from empty, so capacity 1 holds nothing");
        }
    }

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    /// Producer side. Returns false if the queue is full.
    bool push(const T& item) {
        const size_t current_tail = tail.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask;

        if (next_tail == cached_head) {
            // Cached view says full; take the cost of a real load to confirm.
            cached_head = head.load(std::memory_order_acquire);
            if (next_tail == cached_head) {
                return false;
            }
        }

        buffer[current_tail] = item;
        // Release: the item must be visible before the consumer can observe
        // the advanced tail.
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    /// Consumer side. Returns false if the queue is empty.
    bool pop(T& item) {
        const size_t current_head = head.load(std::memory_order_relaxed);

        if (current_head == cached_tail) {
            cached_tail = tail.load(std::memory_order_acquire);
            if (current_head == cached_tail) {
                return false;
            }
        }

        item = buffer[current_head];
        head.store((current_head + 1) & mask, std::memory_order_release);
        return true;
    }

    /// Total slots including the reserved one; always a power of two.
    size_t capacity() const { return buffer.size(); }

    /// Maximum items the queue can actually hold.
    size_t usable_capacity() const { return buffer.size() - 1; }

    /// Approximate occupancy. Exact only when the other side is quiescent.
    size_t size() const {
        const size_t t = tail.load(std::memory_order_acquire);
        const size_t h = head.load(std::memory_order_acquire);
        return (t - h) & mask;
    }

    bool empty() const { return size() == 0; }
};
