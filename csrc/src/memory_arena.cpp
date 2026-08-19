#include "../include/memory_arena.hpp"
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

MemoryArena::MemoryArena(size_t size) : buffer(nullptr), capacity(size), offset(0) {
    if (capacity == 0) {
        throw std::invalid_argument("MemoryArena: capacity must be non-zero");
    }

    // posix_memalign, not malloc: malloc only promises 16-byte alignment, which
    // would leave the first block straddling a cache line and silently void the
    // alignment guarantee every downstream array depends on.
    void* raw = nullptr;
    if (posix_memalign(&raw, ALIGNMENT, capacity) != 0) {
        throw std::bad_alloc();
    }
    buffer = static_cast<uint8_t*>(raw);
}

MemoryArena::~MemoryArena() {
    std::free(buffer); // posix_memalign blocks are released with free()
}

void* MemoryArena::allocate(size_t size) {
    size_t aligned_size = (size + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);

    if (aligned_size < size) { // rounding wrapped around
        throw std::runtime_error("MemoryArena: allocation size overflow");
    }
    if (aligned_size > capacity - offset) {
        throw std::runtime_error(
            "MemoryArena out of memory: requested " + std::to_string(aligned_size) +
            " bytes, only " + std::to_string(capacity - offset) + " of " +
            std::to_string(capacity) + " remain. Increase arena_bytes.");
    }

    void* ptr = buffer + offset;
    offset += aligned_size;
    return ptr;
}

void MemoryArena::reset() {
    offset = 0;
}

size_t MemoryArena::get_capacity() const {
    return capacity;
}

size_t MemoryArena::get_used() const {
    return offset;
}

size_t MemoryArena::get_remaining() const {
    return capacity - offset;
}
