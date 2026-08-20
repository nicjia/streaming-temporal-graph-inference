#pragma once
#include <cstdint>

constexpr uint32_t EMPTY_GAP = 0xFFFFFFFF;

// 8-byte aligned structure for cache efficiency
struct alignas(8) TemporalEdge {
    uint32_t target_node; // Target vertex ID (or EMPTY_GAP)
    uint32_t timestamp;   // Unix epoch timestamp (uint32_t)
};

// Relation type, stored in a parallel array rather than widened into
// TemporalEdge.
//
// Struct-of-arrays, deliberately. Fattening the edge record to carry a type
// would cut the number of edges per 64-byte cache line from eight to five or
// four, and every adjacency scan would pay that -- including the many that only
// need targets and timestamps. A parallel byte array costs 12.5% more memory
// and is touched only by readers that actually want relations.
// uint16_t, not uint8_t: 256 relation types is a real ceiling for a
// general-purpose library. Full CAMEO event codes run to roughly 2,000, SEC
// 8-K item codes and exchange message types are similarly sized. Two bytes per
// edge against an 8-byte edge record is a 25% overhead on a side array that
// only relation-aware readers touch at all.
using EdgeRelation = uint16_t;

constexpr EdgeRelation RELATION_UNKNOWN = 0;