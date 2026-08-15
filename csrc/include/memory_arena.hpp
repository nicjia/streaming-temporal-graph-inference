#pragma once
#include <cstddef>
#include <cstdint>

class MemoryArena {
    uint8_t* buffer;
    size_t capacity;
    size_t offset;

public:
    explicit MemoryArena(size_t size);
    ~MemoryArena();

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    void* allocate(size_t size);
    void reset();
    size_t get_capacity() const;
    size_t get_used() const;
};