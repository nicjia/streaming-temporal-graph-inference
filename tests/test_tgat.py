"""
Tests for the temporal model stack: time encoding, PCSR sampling, attention,
and end-to-end link prediction.

The ones that matter most are the leakage tests. A temporal model that
accidentally sees the future scores beautifully and is worthless, and nothing
about the loss curve reveals it -- it has to be tested for directly.

Run: python tests/test_tgat.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from models import (  # noqa: E402
    PCSRTemporalSampler,
    TemporalAttentionLayer,
    TGAT,
    TGATLinkModel,
    TimeEncode,
)

FAILURES = []


def check(condition, what):
    if condition:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        FAILURES.append(what)


def build_graph(src, dst, ts, num_nodes, slots_per_edge=4):
    graph = graph_engine.PCSRGraph(num_nodes, max(len(src) * slots_per_edge, num_nodes * 8),
                                   256 * 1024 * 1024)
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32), ts.astype(np.uint32))
    return graph


def community_stream(num_nodes=200, num_communities=4, num_edges=40_000, seed=0):
    """
    Synthetic stream with a signal worth learning: nodes interact mostly inside
    their community. A model that has learned anything should rank a real
    destination above a random one.
    """
    rng = np.random.default_rng(seed)
    community = np.arange(num_nodes) % num_communities

    src = rng.integers(0, num_nodes, num_edges)
    same = rng.random(num_edges) < 0.9
    dst = np.empty(num_edges, dtype=np.int64)
    for i in range(num_edges):
        if same[i]:
            candidates = np.flatnonzero(community == community[src[i]])
        else:
            candidates = np.flatnonzero(community != community[src[i]])
        dst[i] = candidates[rng.integers(len(candidates))]

    ts = np.sort(rng.integers(1_600_000_000, 1_600_086_400, num_edges))
    return src.astype(np.int64), dst, ts.astype(np.int64), num_nodes


# ---------------------------------------------------------------------------

def test_time_encoding():
    print("\nTimeEncode (Bochner)")
    enc = TimeEncode(64)

    check(enc(torch.zeros(5)).shape == (5, 64), "output width matches `dimension`")
    check(enc(torch.zeros(3, 7)).shape == (3, 7, 64), "leading dims are preserved")

    try:
        TimeEncode(63)
        check(False, "odd dimension is rejected")
    except ValueError:
        check(True, "odd dimension is rejected")

    # Monte-Carlo normalisation: <Phi(t), Phi(t)> == 1 for any t.
    t = torch.tensor([0.0, 1.0, 3600.0, 1e6])
    norms = (enc(t) * enc(t)).sum(-1)
    check(torch.allclose(norms, torch.ones(4), atol=1e-4), "encodings are unit norm")

    # Translation invariance: the kernel must depend only on the difference.
    a = torch.tensor([0.0, 5_000.0, 900_000.0])
    gap = 137.0
    kernel = (enc(a) * enc(a + gap)).sum(-1)
    check(float((kernel - kernel[0]).abs().max()) < 1e-3,
          "inner product depends only on the time difference")

    # Distinguishes direction, which a cos-only encoding cannot.
    forward = enc(torch.tensor([50.0]))
    backward = enc(torch.tensor([-50.0]))
    check(not torch.allclose(forward, backward),
          "Phi(+dt) differs from Phi(-dt) (sin half is present)")

    # Different timescales must be separable, not aliased into each other.
    minute, hour, day = enc(torch.tensor([60.0])), enc(torch.tensor([3600.0])), enc(torch.tensor([86400.0]))
    check(float((minute * hour).sum()) < 0.999 and float((hour * day).sum()) < 0.999,
          "distinct timescales get distinct encodings")

    loss = enc(torch.tensor([10.0, 100.0])).sum()
    loss.backward()
    check(enc.frequencies.grad is not None and float(enc.frequencies.grad.abs().sum()) > 0,
          "frequencies are learnable")


def test_sampler_causality():
    print("\nPCSRTemporalSampler")
    rng = np.random.default_rng(7)
    num_nodes, num_edges = 300, 30_000
    src = rng.integers(0, num_nodes, num_edges)
    dst = rng.integers(0, num_nodes, num_edges)
    ts = np.sort(rng.integers(1_000, 50_000, num_edges))

    # Tiny initial capacity so the graph grows and the views must be re-acquired.
    graph = graph_engine.PCSRGraph(num_nodes, 64, 256 * 1024 * 1024)
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32), ts.astype(np.uint32))
    check(graph.resize_count > 0, "test exercises PMA growth")

    sampler = PCSRTemporalSampler(graph)
    check(sampler.validate(), "adjacency runs are ascending in time")

    queries = rng.integers(0, num_nodes, 400)
    query_times = rng.integers(1_000, 50_000, 400)
    K = 12
    ids, times, mask = sampler.sample(queries, query_times, K)

    check(ids.shape == (400, K) and mask.shape == (400, K), "fixed output shape")
    check(bool((times[mask] < np.repeat(query_times[:, None], K, 1)[mask]).all()),
          "no sampled event is at or after the query time")

    # Exhaustive cross-check against the obvious slow implementation.
    mismatches = 0
    for i, (v, t) in enumerate(zip(queries, query_times)):
        eligible = np.sort(ts[(src == v) & (ts < t)])[-K:]
        got = np.sort(times[i][mask[i]])
        if not np.array_equal(eligible, got):
            mismatches += 1
    check(mismatches == 0, f"matches brute force on all 400 queries ({mismatches} off)")

    # Neighbour ids in masked slots must be real vertices.
    check(bool((ids[mask] < num_nodes).all()), "sampled ids are valid vertices")
    check(int((~mask).sum()) > 0, "test covers partially-empty histories")

    # Cold start: querying before the stream begins yields nothing at all.
    cold_ids, _, cold_mask = sampler.sample(queries[:20], np.zeros(20, dtype=np.int64), K)
    check(not cold_mask.any(), "queries before the first event are fully masked")

    # Uniform strategy stays causal too.
    u_ids, u_times, u_mask = sampler.sample(queries, query_times, K,
                                            strategy="uniform",
                                            rng=np.random.default_rng(1))
    check(bool((u_times[u_mask] < np.repeat(query_times[:, None], K, 1)[u_mask]).all()),
          "uniform sampling is causal")

    # A grown PMA reseats its buffers; a stale view would silently read the
    # abandoned block.
    before = graph.edge_capacity
    more = rng.integers(0, num_nodes, 200_000)
    graph.insert_edges(more.astype(np.uint32), more.astype(np.uint32),
                       np.full(200_000, 60_000, dtype=np.uint32))
    check(graph.edge_capacity > before, "further inserts grew the PMA again")
    sampler.refresh()
    check(sampler.validate(), "refresh() picks up the reseated buffers")


def test_attention_layer():
    print("\nTemporalAttentionLayer")
    torch.manual_seed(0)
    layer = TemporalAttentionLayer(node_dim=16, time_dim=8, out_dim=16, num_heads=2,
                                   dropout=0.0).eval()

    B, K = 5, 4
    target = torch.randn(B, 16)
    target_time = torch.randn(B, 8)
    neighbors = torch.randn(B, K, 16)
    neighbor_time = torch.randn(B, K, 8)
    mask = torch.ones(B, K, dtype=torch.bool)

    out = layer(target, target_time, neighbors, neighbor_time, mask)
    check(out.shape == (B, 16), "output shape is (batch, out_dim)")

    # The all-masked row is the cold-start case; it must not produce NaN.
    empty_mask = mask.clone()
    empty_mask[2] = False
    out_empty = layer(target, target_time, neighbors, neighbor_time, empty_mask)
    check(not torch.isnan(out_empty).any(), "fully-masked rows do not produce NaN")

    # Masked slots must be genuinely ignored: scrambling them changes nothing.
    partial = mask.clone()
    partial[:, 2:] = False
    scrambled = neighbors.clone()
    scrambled[:, 2:] = torch.randn(B, K - 2, 16) * 100
    a = layer(target, target_time, neighbors, neighbor_time, partial)
    b = layer(target, target_time, scrambled, neighbor_time, partial)
    check(torch.allclose(a, b, atol=1e-5), "masked neighbours have no influence")

    try:
        TemporalAttentionLayer(16, 8, 15, num_heads=2)
        check(False, "out_dim not divisible by num_heads is rejected")
    except ValueError:
        check(True, "out_dim not divisible by num_heads is rejected")


def test_no_future_leakage():
    print("\nEnd-to-end causality")
    rng = np.random.default_rng(3)
    num_nodes = 100
    src = rng.integers(0, num_nodes, 5_000)
    dst = rng.integers(0, num_nodes, 5_000)
    ts = np.sort(rng.integers(1_000, 20_000, 5_000))

    split = 10_000  # embeddings queried at this time must ignore everything after

    past_only = ts < split
    graph_past = build_graph(src[past_only], dst[past_only], ts[past_only], num_nodes)
    graph_all = build_graph(src, dst, ts, num_nodes)

    torch.manual_seed(0)
    model_past = TGAT(num_nodes, PCSRTemporalSampler(graph_past), node_dim=16,
                      time_dim=16, num_layers=2, num_neighbors=8, dropout=0.0).eval()
    torch.manual_seed(0)
    model_all = TGAT(num_nodes, PCSRTemporalSampler(graph_all), node_dim=16,
                     time_dim=16, num_layers=2, num_neighbors=8, dropout=0.0).eval()

    query_nodes = np.arange(num_nodes)
    query_times = np.full(num_nodes, split, dtype=np.int64)

    with torch.no_grad():
        emb_past = model_past(query_nodes, query_times)
        emb_all = model_all(query_nodes, query_times)

    check(torch.allclose(emb_past, emb_all, atol=1e-5),
          "embeddings at time t are identical whether or not future edges exist")

    with torch.no_grad():
        again = model_all(query_nodes, query_times)
    check(torch.allclose(emb_all, again), "eval-mode forward is deterministic")


def test_link_prediction_learns():
    print("\nLink prediction on a planted signal")
    src, dst, ts, num_nodes = community_stream()

    # Temporal split: train on the past, evaluate on the future. A random split
    # would let the model train on events that postdate what it is tested on.
    cutoff = int(len(ts) * 0.8)
    graph = build_graph(src[:cutoff], dst[:cutoff], ts[:cutoff], num_nodes)
    sampler = PCSRTemporalSampler(graph)

    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    model = TGATLinkModel(num_nodes, sampler, node_dim=32, time_dim=32,
                          num_layers=2, num_neighbors=10, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def auc(scores_pos, scores_neg):
        # Rank-based AUC: P(positive scored above negative).
        combined = np.concatenate([scores_pos, scores_neg])
        order = combined.argsort().argsort() + 1
        n_pos, n_neg = len(scores_pos), len(scores_neg)
        rank_sum = order[:n_pos].sum()
        return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    def evaluate():
        model.eval()
        pos, neg = [], []
        with torch.no_grad():
            for start in range(cutoff, min(cutoff + 2_000, len(ts)), 200):
                end = min(start + 200, len(ts))
                s, d, t = src[start:end], dst[start:end], ts[start:end]
                negatives = rng.integers(0, num_nodes, end - start)
                pos.append(model.score(s, d, t).numpy())
                neg.append(model.score(s, negatives, t).numpy())
        return auc(np.concatenate(pos), np.concatenate(neg))

    start_auc = evaluate()

    model.train()
    batch_size = 200
    # Start past the first events so the roots have some history to attend to.
    begin = cutoff // 4
    losses = []
    for _ in range(4):
        for offset in range(begin, cutoff, batch_size):
            end = min(offset + batch_size, cutoff)
            s, d, t = src[offset:end], dst[offset:end], ts[offset:end]
            negatives = rng.integers(0, num_nodes, end - offset)

            optimizer.zero_grad()
            loss, _, _ = model.loss(s, d, t, negatives)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    end_auc = evaluate()
    first, last = np.mean(losses[:5]), np.mean(losses[-5:])

    print(f"       loss {first:.4f} -> {last:.4f} over {len(losses)} steps")
    print(f"       held-out AUC {start_auc:.3f} -> {end_auc:.3f}")

    check(last < first, "training loss decreases")
    check(end_auc > start_auc, "held-out AUC improves over the untrained model")
    check(end_auc > 0.65, f"held-out AUC clears 0.65 (got {end_auc:.3f})")


def main():
    test_time_encoding()
    test_sampler_causality()
    test_attention_layer()
    test_no_future_leakage()
    test_link_prediction_learns()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All model tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
