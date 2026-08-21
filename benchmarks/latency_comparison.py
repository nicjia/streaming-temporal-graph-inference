"""
Edge-insertion latency: PCSR (C++) vs NetworkX vs a plain Python dict-of-lists.

Three things this is careful about, because the easy version of this benchmark
flatters the C++ engine for the wrong reasons:

1. The workload is power-law, not uniform. Uniform endpoints let the PMA stay
   almost entirely on its O(1) fast path, so it measures an array write rather
   than a dynamic graph structure. Real GDELT is heavily skewed, and skew is
   what makes rebalancing happen at all. Pass --uniform to see the gap.

2. NetworkX is not the only baseline. A MultiDiGraph carries per-edge attribute
   dicts the PCSR has no equivalent for, so beating it partly measures features
   nobody asked for. `dict-of-lists` is the honest floor: the least Python you
   could write and still have a queryable temporal adjacency structure.

3. Correctness is asserted, not assumed. Every structure is scanned afterwards
   and must report the same edge count. A structure that drops edges can be
   arbitrarily fast.

Usage:
    python benchmarks/latency_comparison.py                 # synthetic, 200k edges
    python benchmarks/latency_comparison.py --edges 1000000
    python benchmarks/latency_comparison.py --gdelt data/gdelt/gdelt_*.csv
    python benchmarks/latency_comparison.py --uniform
"""

import argparse
import gc
import glob
import os
import statistics
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_engine  # noqa: E402

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import psutil
except ImportError:
    psutil = None


# --------------------------------------------------------------------------
# Workload generation
# --------------------------------------------------------------------------

def make_zipf_workload(num_nodes, num_edges, alpha=1.1, seed=42):
    """Power-law sources, uniform targets -- the GDELT shape."""
    rng = np.random.default_rng(seed)

    ranks = np.arange(1, num_nodes + 1, dtype=np.float64)
    weights = 1.0 / ranks ** alpha
    weights /= weights.sum()

    src = rng.choice(num_nodes, size=num_edges, p=weights).astype(np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    ts = np.sort(rng.integers(1_600_000_000, 1_600_086_400, size=num_edges)).astype(np.uint32)
    return src, dst, ts


def make_uniform_workload(num_nodes, num_edges, seed=42):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    dst = rng.integers(0, num_nodes, size=num_edges, dtype=np.uint32)
    ts = np.sort(rng.integers(1_600_000_000, 1_600_086_400, size=num_edges)).astype(np.uint32)
    return src, dst, ts


def load_gdelt_workload(pattern, repeat=1):
    """Real GDELT events. Repeats the slice to reach a measurable edge count."""
    import pandas as pd

    paths = sorted(glob.glob(pattern))
    if pattern.endswith(".csv"):
        # Slices are stored gzipped; accept both so a plain-.csv glob still
        # finds everything.
        paths = sorted(set(paths) | set(glob.glob(pattern + ".gz")))
    if not paths:
        raise SystemExit(f"No GDELT CSVs matched {pattern!r}")

    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Actor1Name", "Actor2Name", "DATEADDED"])

    actors = pd.unique(
        pd.concat([df["Actor1Name"].str.strip().str.upper(),
                   df["Actor2Name"].str.strip().str.upper()])
    )
    index = {name: i for i, name in enumerate(actors)}

    src = df["Actor1Name"].str.strip().str.upper().map(index).to_numpy(dtype=np.uint32)
    dst = df["Actor2Name"].str.strip().str.upper().map(index).to_numpy(dtype=np.uint32)
    ts = (pd.to_datetime(df["DATEADDED"], format="%Y%m%d%H%M%S").astype("int64") // 10**9)
    ts = ts.to_numpy(dtype=np.uint32)

    if repeat > 1:
        src = np.tile(src, repeat)
        dst = np.tile(dst, repeat)
        ts = np.tile(ts, repeat)

    order = np.argsort(ts, kind="stable")
    return src[order], dst[order], ts[order], len(actors), len(paths)


def describe(src, num_nodes):
    degrees = np.bincount(src, minlength=num_nodes)
    top = max(1, num_nodes // 100)
    share = np.sort(degrees)[::-1][:top].sum() / len(src)
    return (f"{len(src):,} edges over {num_nodes:,} vertices | "
            f"max degree {degrees.max():,}, median {int(np.median(degrees))}, "
            f"top 1% hold {share:.1%}")


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def rss_mb():
    if psutil is None:
        return float("nan")
    return psutil.Process().memory_info().rss / (1024 * 1024)


class Result:
    def __init__(self, name, seconds, edges, retained, latencies=None, mem_mb=None,
                 scan_seconds=None, note=""):
        self.name = name
        self.seconds = seconds
        self.edges = edges
        self.retained = retained
        self.latencies = latencies
        self.mem_mb = mem_mb
        self.scan_seconds = scan_seconds
        self.note = note

    @property
    def throughput(self):
        return self.edges / self.seconds

    @property
    def ns_per_edge(self):
        return self.seconds * 1e9 / self.edges


def sample_latencies(insert_fn, src, dst, ts, sample_every=97):
    """
    Per-op timing on a strided sample.

    Timing every insert would swamp a ~15 ns operation with ~40 ns of clock
    overhead. A stride keeps the overhead bounded while still catching the
    rebalance spikes, which is the tail that matters for a streaming consumer.
    The prime stride avoids aliasing with any periodicity in the workload.
    """
    out = []
    perf = time.perf_counter_ns
    for i in range(0, len(src), sample_every):
        a = perf()
        insert_fn(int(src[i]), int(dst[i]), int(ts[i]))
        out.append(perf() - a)
    return out


def provision(num_edges, num_nodes, slack, slots_per_vertex):
    """
    Pick a PMA capacity.

    Two independent floors. `slack` covers the total edge count -- a PMA that
    is 100% full has no gaps left to insert into. `slots_per_vertex` covers the
    per-vertex regions, which matters more than it looks: a graph with far more
    vertices than the slack alone would cover gives each region only a handful
    of slots, so even modest degrees overflow and rebalance constantly. Under-
    provisioning does not lose edges any more (that bug is fixed), it just
    quietly multiplies the work per insert -- watch the slots-rewritten note.
    """
    return max(int(num_edges * slack), num_nodes * slots_per_vertex)


def bench_pcsr(src, dst, ts, num_nodes, capacity):
    # Growth re-allocates from the arena without reclaiming old blocks, so the
    # budget is deliberately generous relative to the live footprint.
    arena = max(256 * 1024 * 1024, capacity * 8 * 4)

    gc.collect()
    before = rss_mb()
    graph = graph_engine.PCSRGraph(num_nodes, capacity, arena)

    start = time.perf_counter()
    for i in range(len(src)):
        graph.insert_edge(int(src[i]), int(dst[i]), int(ts[i]))
    seconds = time.perf_counter() - start
    mem = rss_mb() - before

    scan_start = time.perf_counter()
    counts = np.asarray(graph.get_vertex_counts())
    retained = int(counts.sum())
    scan_seconds = time.perf_counter() - scan_start

    fresh = graph_engine.PCSRGraph(num_nodes, capacity, arena)
    latencies = sample_latencies(fresh.insert_edge, src, dst, ts)

    note = (f"{graph.rebalance_count:,} rebalances, {graph.resize_count} resizes, "
            f"{graph.slots_rewritten / len(src):.2f} slots rewritten/edge")
    return Result("PCSR (C++), per call", seconds, len(src), retained, latencies, mem,
                  scan_seconds, note)


def bench_pcsr_bulk(src, dst, ts, num_nodes, capacity, batch_size):
    """
    The same engine, fed whole batches instead of one edge per Python call.

    This is the row that actually tests the architecture. Everything else in
    this file is bottlenecked on the interpreter: boxing two ints and
    dispatching through pybind costs microseconds, while the insert itself is
    nanoseconds. Streaming a batch of numpy buffers crosses the boundary once
    and runs with the GIL released.
    """
    arena = max(256 * 1024 * 1024, capacity * 8 * 4)

    gc.collect()
    before = rss_mb()
    graph = graph_engine.PCSRGraph(num_nodes, capacity, arena)

    batch_latencies = []
    perf = time.perf_counter_ns
    start = time.perf_counter()
    for begin in range(0, len(src), batch_size):
        end = begin + batch_size
        a = perf()
        graph.insert_edges(src[begin:end], dst[begin:end], ts[begin:end])
        batch_latencies.append(perf() - a)
    seconds = time.perf_counter() - start
    mem = rss_mb() - before

    retained = int(graph.num_edges)

    note = (f"batches of {batch_size:,}; "
            f"{graph.rebalance_count:,} rebalances, {graph.resize_count} resizes, "
            f"{graph.slots_rewritten / len(src):.2f} slots rewritten/edge")
    result = Result("PCSR (C++), bulk", seconds, len(src), retained, None, mem,
                    None, note)
    result.batch_latencies = batch_latencies
    result.batch_size = batch_size
    return result


def bench_networkx(src, dst, ts):
    if nx is None:
        return None

    gc.collect()
    before = rss_mb()
    graph = nx.MultiDiGraph()

    start = time.perf_counter()
    for i in range(len(src)):
        graph.add_edge(int(src[i]), int(dst[i]), timestamp=int(ts[i]))
    seconds = time.perf_counter() - start
    mem = rss_mb() - before

    scan_start = time.perf_counter()
    retained = graph.number_of_edges()
    scan_seconds = time.perf_counter() - scan_start

    fresh = nx.MultiDiGraph()
    latencies = sample_latencies(
        lambda s, d, t: fresh.add_edge(s, d, timestamp=t), src, dst, ts)

    return Result("NetworkX MultiDiGraph", seconds, len(src), retained, latencies,
                  mem, scan_seconds, "per-edge attribute dicts")


def bench_dict_of_lists(src, dst, ts):
    gc.collect()
    before = rss_mb()
    adjacency = defaultdict(list)

    start = time.perf_counter()
    for i in range(len(src)):
        adjacency[int(src[i])].append((int(dst[i]), int(ts[i])))
    seconds = time.perf_counter() - start
    mem = rss_mb() - before

    scan_start = time.perf_counter()
    retained = sum(len(v) for v in adjacency.values())
    scan_seconds = time.perf_counter() - scan_start

    fresh = defaultdict(list)
    latencies = sample_latencies(
        lambda s, d, t: fresh[s].append((d, t)), src, dst, ts)

    return Result("Python dict-of-lists", seconds, len(src), retained, latencies,
                  mem, scan_seconds, "no gap management, append-only")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report(results, expected_edges):
    baseline = next((r for r in results if r.name == "PCSR (C++), per call"), None)

    print()
    print(f"{'structure':<24} {'insert (s)':>11} {'edges/s':>13} {'ns/edge':>10} "
          f"{'speedup':>9} {'RSS MB':>8} {'edges kept':>12}")
    print("-" * 94)
    for r in results:
        speedup = (r.seconds / baseline.seconds) if baseline and r is not baseline else 1.0
        kept = f"{r.retained:,}"
        if r.retained != expected_edges:
            kept += " !!"
        print(f"{r.name:<24} {r.seconds:>11.4f} {r.throughput:>13,.0f} "
              f"{r.ns_per_edge:>10.1f} {speedup:>8.1f}x {r.mem_mb:>8.1f} {kept:>12}")

    print()
    print(f"{'structure':<24} {'p50 ns':>10} {'p90 ns':>10} {'p99 ns':>10} "
          f"{'p99.9 ns':>11} {'max ns':>12}")
    print("-" * 82)
    for r in results:
        if not r.latencies:
            continue
        s = sorted(r.latencies)
        def pct(p):
            return s[min(len(s) - 1, int(p * len(s)))]
        print(f"{r.name:<24} {pct(0.50):>10,} {pct(0.90):>10,} {pct(0.99):>10,} "
              f"{pct(0.999):>11,} {max(s):>12,}")

    for r in results:
        batch = getattr(r, "batch_latencies", None)
        if not batch:
            continue
        b = sorted(batch)
        def bpct(p):
            return b[min(len(b) - 1, int(p * len(b)))]
        print()
        print(f"  {r.name}: per-batch latency over {len(b):,} batches of "
              f"{r.batch_size:,} edges")
        print(f"    p50 {bpct(0.50) / 1000:,.1f} us   p99 {bpct(0.99) / 1000:,.1f} us   "
              f"max {max(b) / 1000:,.1f} us   "
              f"({statistics.mean(b) / r.batch_size:,.1f} ns/edge amortised)")

    print()
    for r in results:
        if r.note:
            print(f"  {r.name}: {r.note}")

    print()
    mismatched = [r for r in results if r.retained != expected_edges]
    if mismatched:
        print("FAIL: these structures did not retain every edge:")
        for r in mismatched:
            print(f"  {r.name}: kept {r.retained:,} of {expected_edges:,}")
        return 1
    print(f"All structures retained {expected_edges:,} edges.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edges", type=int, default=200_000)
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=1.1,
                        help="Zipf exponent; higher means more skew")
    parser.add_argument("--uniform", action="store_true",
                        help="Use uniform endpoints instead of power-law")
    parser.add_argument("--gdelt", metavar="GLOB",
                        help="Replay real GDELT CSVs instead of synthetic data")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Tile the GDELT slice N times for a longer run")
    parser.add_argument("--skip-networkx", action="store_true")
    parser.add_argument("--slack", type=float, default=2.0,
                        help="PMA slots per edge (1.0 = no gaps at all)")
    parser.add_argument("--slots-per-vertex", type=int, default=16,
                        help="Minimum PMA slots per vertex region")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Edges per bulk insert call")
    args = parser.parse_args()

    if args.gdelt:
        src, dst, ts, num_nodes, files = load_gdelt_workload(args.gdelt, args.repeat)
        source = f"GDELT ({files} file(s), repeat={args.repeat})"
    elif args.uniform:
        num_nodes = args.nodes
        src, dst, ts = make_uniform_workload(num_nodes, args.edges)
        source = "synthetic uniform"
    else:
        num_nodes = args.nodes
        src, dst, ts = make_zipf_workload(num_nodes, args.edges, args.alpha)
        source = f"synthetic zipf(alpha={args.alpha})"

    print(f"Workload: {source}")
    print(f"          {describe(src, num_nodes)}")
    if psutil is None:
        print("          (psutil not installed -- RSS column will be nan)")

    capacity = provision(len(src), num_nodes, args.slack, args.slots_per_vertex)
    print(f"          PMA provisioned with {capacity:,} slots "
          f"({capacity / len(src):.2f} per edge, {capacity / num_nodes:.1f} per vertex)")

    results = [
        bench_pcsr(src, dst, ts, num_nodes, capacity),
        bench_pcsr_bulk(src, dst, ts, num_nodes, capacity, args.batch_size),
    ]
    if not args.skip_networkx:
        nx_result = bench_networkx(src, dst, ts)
        if nx_result is None:
            print("          (networkx not installed -- skipping)")
        else:
            results.append(nx_result)
    results.append(bench_dict_of_lists(src, dst, ts))

    return report(results, len(src))


if __name__ == "__main__":
    sys.exit(main())
