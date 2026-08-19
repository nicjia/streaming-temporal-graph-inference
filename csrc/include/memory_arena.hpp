#pragma once
#include <cstddef>
#include <cstdint>

/**
 * @brief Bump allocator over one pre-faulted, cache-line-aligned block.
 *
 * Every block handed out by allocate() starts on a 64-byte boundary, so two
 * hot arrays can never share a cache line (no false sharing between them).
 * That requires BOTH a 64-byte aligned base pointer (plain malloc only
 * guarantees 16) and 64-byte rounding of every allocation size.
 *
 * There is no per-block free: memory is reclaimed only by reset() or by
 * destroying the arena. Callers that re-allocate a growing buffer therefore
 * abandon the old block, so size the arena for the sum of all allocations,
 * not the live set.
 */
class MemoryArena {
    uint8_t* buffer;
    size_t capacity;
    size_t offset;

public:
    static constexpr size_t ALIGNMENT = 64; // x86-64 / Apple Silicon cache line

    explicit MemoryArena(size_t size);
    ~MemoryArena();

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;

    /// Returns a 64-byte aligned block of at least `size` bytes.
    /// Throws std::runtime_error (naming the shortfall) if the arena is full.
    void* allocate(size_t size);
    void reset();
    size_t get_capacity() const;
    size_t get_used() const;
    size_t get_remaining() const;
};
