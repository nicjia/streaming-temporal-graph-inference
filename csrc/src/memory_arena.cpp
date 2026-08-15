#include "../include/memory_arena.hpp"
#include <cstdlib>
#include <stdexcept>

MemoryArena::MemoryArena(size_t size) : capacity(size), offset(0) {
    buffer = static_cast<uint8_t*>(std::malloc(capacity));
    if (!buffer) {
        throw std::bad_alloc();
    }
}

MemoryArena::~MemoryArena() {
    if (buffer) {
        std::free(buffer);
    }
}

void* MemoryArena::allocate(size_t size) {
    size_t aligned_size = (size + 7) & ~7; 
    
    if (offset + aligned_size > capacity) {
        throw std::runtime_error("Arena Out of Memory - Size limit reached");
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