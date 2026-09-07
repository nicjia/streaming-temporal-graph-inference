# Streaming Temporal Graph Inference

This is a system to build a temporal graph out of streamed data while learning on it. It relies on several parts.

The central graph core is written in C++20, and the graph is stored as a Packed Compressed Sparse Row (PCSR) over a Packed Memory Array. This structure takes in edges through a lock-free queue and hands adjacency to Python as zero-copy NumPy views. More on the data structures below.

The edges are processed by turning SEC 8-K filings into typed, timestamped edges using a local small language model (SLM) with constrained decoding, so that the model can only produce restricted-vocabulary labels.

The temporal graph attention network (TGAT) is trained on the resulting corporate graph and evaluated on next-relationship prediction.

## Results
The graph has 131,079 edges over 10,081 companies, combined from three ticker-keyed sources: 8-K text relations, director interlocks (SEC Form 3/4/5), and FactSet Revere supplier–customer links.

### Next-Relationship Prediction
Given the network state at time *t*, rank the firm each source connects to next against 50 degree-matched negatives. Reported on new pairs:

| scorer | AUC | Recall@10 |
|---|---|---|
| Adamic–Adar | 0.567 | 0.27 |
| **TGAT** | **0.68** | **0.37** |

### Ablation Study
Retraining under different constraints and removing one component at a time:

| variant | AUC |
|---|---|
| topology + time | 0.559 |
| + relation types | **0.689** |
| one hop only | 0.668 |

The bonus comes from **relation types**, specifically the semantics derived from filing text, not from the shape of the graph.

### Returns: null
Attempts to turn the graph into a daily return signal are null across different studies. This architecture is strong for forecasting corporate events and structure, but not market returns.

## System Details
The core of the system is a dynamic graph that remains fast and agile under continuous streams of edge insertions and expirations.

- **PCSR over a Packed Memory Array.** Unlike traditional adjacency lists, which either scatter across memory or require reallocation on edge insertion, a PCSR gives each node a buffer of spare room, so most edge insertions are a direct write. When a node's allocated area fills up, a nearby window is rebalanced to redistribute the free space. An offsets array stores the start index of each node's adjacency list, so any node's neighbors can be located in O(1).
- **Lock-free ingestion.** A single-producer/single-consumer ring buffer feeds edges from the producer thread into the graph without locks.
- **Zero-copy Python bridge.** Pybind11 exposes the internal structure's offset, count, and edge arrays as NumPy views without any copying, so the sampler can operate directly on the packed memory.
- **Edge expiration.** Expiration drops the timestamp-ordered prefix in amortized O(1) using a per-vertex start offset.
- **Typed, weighted edges.** Edges can carry relation type, model confidence, and source provenance. Edges can also optionally carry a weight (off by default), which is projected into attention so that event magnitude can affect both which neighbors are attended to and the value passed forward.
- **Throughput.** ~35 ns per edge insert.

## Data Used
This project used SEC filings pulled from SEC EDGAR. A closed-vocabulary schema of relation type, sign, magnitude bucket, and confidence was enforced by GBNF grammar, so the model's output is always in-vocabulary and the numeric edge weight is derived from the labels deterministically. This runs on Qwen2.5-7B on CPU.

Listed equities were resolved by ticker, and everything else was resolved by normalizing string identity.

Despite using multiple sources, all edges take the form (src, dst, timestamp, source, relation, weight).

## Build & reproduce
```bash
# C++ core + tests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Python extension + model/strategy tests
make ext
python tests/test_tgat.py
python tests/test_strategy.py
```

The graph is persisted, so the studies run without re-extracting:
```bash
python benchmarks/corporate_link_prediction.py   # next-relationship result
python benchmarks/corporate_link_variants.py     # ablation + typed head
python benchmarks/link_prediction_coldstart.py   # the cold-start split
python benchmarks/multisource_propagation.py     # the return null
```

## Layout
```
csrc/             C++20 core: PMA graph, lock-free queue, Pybind11 bindings
python/models/    TGAT, temporal neighbour sampler
python/ingestion/ 8-K fetch, constrained extraction, entity resolution
benchmarks/       graph assembly and the link-prediction studies
tests/            C++ (ctest + sanitizers) and Python tests
```
