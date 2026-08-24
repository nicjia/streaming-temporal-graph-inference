"""
Benchmark A: the systems test, on real Ethereum transactions.

No NLP, no extraction, no model. Raw on-chain edges into the C++ engine, to
measure the three structural claims the project makes: ingestion throughput,
zero-allocation steady state, and per-batch tail latency on the streaming path.

Usage:
  python benchmarks/ethereum_ingest.py --download 6
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import graph_engine  # noqa: E402
from ingestion.ethereum import load_transactions  # noqa: E402

REPO = "vnegi10/Ethereum_blockchain_parquet"


def fetch(num_files):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(REPO, files_metadata=True)
    tx = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("transactions/"))
    bl = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("blocks/"))
    print(f"downloading {num_files} of {len(tx)} shards...")
    for name in tx[-num_files:] + bl[-num_files:]:
        hf_hub_download(REPO, name, repo_type="dataset")


def cached_globs():
    root = os.path.expanduser("~/.cache/huggingface/hub")
    tx = sorted(glob.glob(f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet/snapshots/*/transactions/*.parquet"))
    bl = sorted(glob.glob(f"{root}/datasets--vnegi10--Ethereum_blockchain_parquet/snapshots/*/blocks/*.parquet"))
    if not tx:
        raise SystemExit("no cached shards; run with --download N first")
    return os.path.dirname(tx[0]) + "/*.parquet", os.path.dirname(bl[0]) + "/*.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=int, default=0)
    parser.add_argument("--min-degree", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--queue", type=int, default=65536)
    args = parser.parse_args()

    if args.download:
        fetch(args.download)

    tx_glob, block_glob = cached_globs()
    table, addresses = load_transactions(tx_glob, block_glob, min_degree=args.min_degree)

    src = table["src"].to_numpy(np.uint32)
    dst = table["dst"].to_numpy(np.uint32)
    # Block time fits uint32 until 2106; the engine stores seconds.
    ts = table["ts"].to_numpy(np.int64).astype(np.uint32)
    rel = table["relation"].to_numpy(np.uint16)
    vertices, edges = len(addresses), len(src)

    capacity = int(edges * 2)
    arena = capacity * 8 * 4 + (1 << 26)
    print(f"\nprovisioning {vertices:,} vertices, {capacity:,} slots, {arena/1e9:.2f} GB arena")

    # ---- bulk path -------------------------------------------------------
    graph = graph_engine.PCSRGraph(vertices, capacity, arena)
    start = time.perf_counter()
    graph.insert_edges(src, dst, ts, rel)
    bulk = time.perf_counter() - start
    scanned = int(np.asarray(graph.get_vertex_counts()).sum())

    print(f"\n{'BULK REPLAY':<24}{edges/bulk/1e6:>8.1f} M edges/s{bulk*1e9/edges:>10.1f} ns/edge")
    print(f"{'  lossless':<24}{'yes' if scanned == edges == graph.num_edges else 'NO':>8}"
          f"   ({scanned:,} scanned)")
    print(f"{'  rebalances':<24}{graph.rebalance_count:>8,}   resizes {graph.resize_count}"
          f"   {graph.slots_rewritten/edges:.2f} slots rewritten/edge")
    print(f"{'  arena used':<24}{graph.arena_used/1e6:>8.0f} MB   (allocated once, never grown)")

    # ---- streaming path --------------------------------------------------
    stream_graph = graph_engine.PCSRGraph(vertices, capacity, arena)
    ingestor = graph_engine.StreamingIngestor(stream_graph, args.queue)
    latencies = []
    perf = time.perf_counter_ns

    with ingestor:
        start = time.perf_counter()
        for i in range(0, edges, args.batch):
            j = i + args.batch
            t0 = perf()
            ingestor.push_batch(src[i:j], dst[i:j], ts[i:j], rel[i:j])
            latencies.append(perf() - t0)
        ingestor.drain()
        stream = time.perf_counter() - start

    lat = np.sort(np.array(latencies) / 1e6)  # ms per batch
    per_edge = lat / args.batch * 1e6         # ns per edge within batch
    print(f"\n{'STREAMING (SPSC)':<24}{edges/stream/1e6:>8.1f} M edges/s{stream*1e9/edges:>10.1f} ns/edge")
    print(f"{'  batches':<24}{len(lat):>8,}   of {args.batch:,} edges, queue {ingestor.queue_capacity:,}")
    print(f"{'  batch latency ms':<24}p50 {lat[len(lat)//2]:>6.3f}  p99 {lat[int(len(lat)*.99)]:>6.3f}"
          f"  max {lat[-1]:>6.3f}")
    print(f"{'  per-edge ns':<24}p50 {per_edge[len(per_edge)//2]:>6.0f}  "
          f"p99 {per_edge[int(len(per_edge)*.99)]:>6.0f}")
    print(f"{'  back-pressure':<24}{ingestor.producer_spins:>8,} producer spins, "
          f"{ingestor.consumer_idle_polls:,} consumer idles")
    print(f"{'  lossless':<24}{'yes' if stream_graph.num_edges == edges else 'NO':>8}")

    # ---- read side -------------------------------------------------------
    offsets = np.asarray(graph.get_vertex_offsets())
    counts = np.asarray(graph.get_vertex_counts())
    rng = np.random.default_rng(0)
    probe = rng.integers(0, vertices, 200_000)
    start = time.perf_counter()
    total = int(counts[probe].sum())
    read = time.perf_counter() - start
    print(f"\n{'NEIGHBOURHOOD READ':<24}{200_000/read/1e6:>8.1f} M lookups/s   "
          f"({total/200_000:.1f} mean degree)")
    print(f"\nSub-millisecond p99 per {args.batch:,}-edge batch: "
          f"{'YES' if lat[int(len(lat)*.99)] < 1.0 else 'NO'}")


if __name__ == "__main__":
    main()
