#pragma once
#include <atomic>
#include <vector>
#include <cstddef>

template<typename T>
class SPSCQueue {
    std::vector<T> buffer;
    const size_t capacity;

    alignas(64) std::atomic<size_t> head;
    alignas(64) std::atomic<size_t> tail;

public:
    SPSCQueue(size_t cap) : buffer(cap), capacity(cap), head(0), tail(0) {}

    bool push(const T& item) {
        size_t current_tail = tail.load(std::memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % capacity;
        
        if (next_tail == head.load(std::memory_order_acquire)) {
            return false; 
        }
        
        buffer[current_tail] = item;
        tail.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        size_t current_head = head.load(std::memory_order_relaxed);
        
        if (current_head == tail.load(std::memory_order_acquire)) {
            return false; 
        }
        
        item = buffer[current_head];
        head.store((current_head + 1) % capacity, std::memory_order_release);
        return true;
    }
};