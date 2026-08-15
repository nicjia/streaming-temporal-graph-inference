#pragma once
#include <cstdint>

constexpr uint32_t EMPTY_GAP = 0xFFFFFFFF;

// 8-byte aligned structure for cache efficiency
struct alignas(8) TemporalEdge {
    uint32_t target_node; // Target vertex ID (or EMPTY_GAP)
    uint32_t timestamp;   // Unix epoch timestamp (uint32_t)
};