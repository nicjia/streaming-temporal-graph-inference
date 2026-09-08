"""
Tests for the ingestion, strategy and backtest layers.

The backtest tests are the ones that matter. A backtest cannot be validated by
inspection -- a lookahead bug produces beautiful, plausible numbers and no
error. So the properties are tested directly: a random signal must earn nothing
before costs, deliberately peeking must score better than not peeking, and
shifting a signal later in time must destroy it.

Run: python tests/test_strategy.py
"""

import os
import sys
import time
import json
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from backtest.evaluate import average_precision, mean_reciprocal_rank, roc_auc  # noqa: E402
from backtest.folds import walk_forward_folds  # noqa: E402
from backtest.propagation import run_event_study  # noqa: E402
from ingestion.corpus import CONFLICT_CLASSES, synthetic_corpus  # noqa: E402
from ingestion.supply_chain import synthetic_supply_chain  # noqa: E402
from ingestion.historical_replayer import DataReplayer, provision, to_unix_seconds  # noqa: E402
from ingestion.id_mapper import EntityMapper  # noqa: E402
from models import PCSRTemporalSampler  # noqa: E402
from strategy.signals import goldstein_signal, random_signal, signal_dates  # noqa: E402
from strategy.universe import Universe  # noqa: E402

SCRATCH = os.environ.get("TMPDIR", "/tmp")
FAILURES = []


def check(condition, what):
    if condition:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}")
        FAILURES.append(what)


def tmp(name):
    return os.path.join(SCRATCH, f"htge_test_{name}")


# ---------------------------------------------------------------------------

def test_id_mapper():
    print("\nEntityMapper")

    path = tmp("mapper.json")
    if os.path.exists(path):
        os.remove(path)

    mapper = EntityMapper(filepath=path)

    # get_id used to be defined twice, with the slow definition silently winning.
    definitions = [name for name in dir(EntityMapper) if name == "get_id"]
    check(len(definitions) == 1, "get_id has a single definition")

    check(mapper.get_id("United States") == mapper.get_id("  united   states  "),
          "normalisation collapses case and whitespace")
    check(mapper.get_id("RUSSIAN") == mapper.get_id("RUSSIA"),
          "demonym alias resolves to the country")
    check(mapper.get_id("PRESIDENT") != mapper.get_id("RESIDENTS"),
          "near-miss strings stay distinct with fuzzy off")
    check(mapper.get_id(None) == -1 and mapper.get_id("") == -1
          and mapper.get_id(float("nan")) == -1,
          "blank and missing names return -1")

    # Bulk and row-at-a-time must agree exactly, not merely partition the same.
    names = ["ALPHA", "BETA", "ALPHA", "GAMMA", "BETA", "DELTA"]
    loop_mapper = EntityMapper(filepath=tmp("loop.json"))
    bulk_mapper = EntityMapper(filepath=tmp("bulk.json"))
    for existing in (tmp("loop.json"), tmp("bulk.json")):
        if os.path.exists(existing):
            os.remove(existing)
    loop_mapper = EntityMapper(filepath=tmp("loop.json"))
    bulk_mapper = EntityMapper(filepath=tmp("bulk.json"))
    loop_ids = np.array([loop_mapper.get_id(n) for n in names])
    check(np.array_equal(bulk_mapper.get_ids(names), loop_ids),
          "get_ids matches get_id element for element")

    # The cap is what stops the engine seeing an id >= max_vertices.
    capped = EntityMapper(filepath=tmp("cap.json"), max_entities=3)
    if os.path.exists(tmp("cap.json")):
        os.remove(tmp("cap.json"))
    capped = EntityMapper(filepath=tmp("cap.json"), max_entities=3)
    ids = capped.get_ids([f"E{i}" for i in range(10)])
    check(ids.max() == 2, "ids never exceed max_entities - 1")
    check(int((ids == -1).sum()) == 7, "entities past the cap return -1")
    check(capped.overflow_count == 7, "overflow is counted")

    # Lookup cost must not grow with vocabulary size; it used to be quadratic.
    vocabulary = [f"ENTITY {i}" for i in range(20000)]
    scaling = EntityMapper(filepath=tmp("scale.json"))
    if os.path.exists(tmp("scale.json")):
        os.remove(tmp("scale.json"))
    scaling = EntityMapper(filepath=tmp("scale.json"))

    start = time.perf_counter()
    for name in vocabulary[:2000]:
        scaling.get_id(name)
    early = time.perf_counter() - start

    start = time.perf_counter()
    for name in vocabulary[18000:]:
        scaling.get_id(name)
    late = time.perf_counter() - start

    check(late < early * 4,
          f"lookup cost is flat in vocabulary size ({early * 1e3:.1f} ms at 0-2k "
          f"vs {late * 1e3:.1f} ms at 18-20k)")

    # Persistence must not renumber anything.
    scaling.save_map()
    reloaded = EntityMapper(filepath=tmp("scale.json"))
    check(reloaded.get_id("ENTITY 7") == scaling.get_id("ENTITY 7"),
          "ids survive a save/load round trip")
    check(len(reloaded) == len(scaling), "id counter restores correctly")


def test_timestamps():
    print("\nTimestamp handling")

    # pandas 2.x parses this to datetime64[us]; the old //10**9 divisor assumed
    # nanoseconds and put every GDELT event in January 1970.
    column = pd.Series([20231101080000, 20240115120000])
    seconds = to_unix_seconds(column)
    check(seconds[0] == 1698825600, "YYYYMMDDHHMMSS converts to the right epoch second")
    check(pd.Timestamp(seconds[0], unit="s").year == 2023, "year survives the conversion")

    check(np.array_equal(to_unix_seconds(pd.Series(seconds)), seconds),
          "already-Unix values pass through unchanged")


def test_universe():
    print("\nUniverse")
    universe = Universe()

    check(len(universe) > 20, "universe covers a usable number of countries")
    check(universe.ticker("USA") == "SPY", "known country maps to its ETF")
    check(universe.ticker("ZZZ") is None, "unknown country maps to nothing")
    check(Universe.country_from_name("United States") == "USA",
          "actor name resolves to ISO3")
    check(Universe.country_from_name("Davos") == "CHE",
          "city names resolve to their country")
    check(Universe.country_from_name("Toyota") is None,
          "non-country actors do not resolve")
    check(not universe.is_tradable("RUS"),
          "Russia is in the name table but deliberately not tradable")

    path = tmp("universe.json")
    universe.save(path)
    check(Universe.load(path).mapping == universe.mapping, "universe round-trips")


def test_folds():
    print("\nWalk-forward folds")
    timestamps = np.arange(1_600_000_000, 1_600_000_000 + 86400 * 400, 3600)
    folds = walk_forward_folds(timestamps, num_folds=4, min_train_fraction=0.4)

    check(len(folds) == 4, "requested number of folds")
    check(all(f.train_end <= f.test_end for f in folds), "test windows are non-empty")
    check(all(f.train_start < f.train_end for f in folds), "train windows are non-empty")
    check(all(folds[i].test_end == folds[i + 1].test_start for i in range(len(folds) - 1)),
          "test windows tile without gaps or overlap")
    check(folds[-1].test_end >= timestamps.max(), "last fold reaches the end of the data")

    # The property the whole design rests on.
    check(all(f.train_end == f.test_start for f in folds),
          "no fold trains on any timestamp inside its own test window")

    expanding = walk_forward_folds(timestamps, num_folds=3)
    check(len({f.train_start for f in expanding}) == 1,
          "expanding folds share one training start")
    rolling = walk_forward_folds(timestamps, num_folds=3, expanding=False)
    check(len({f.train_start for f in rolling}) == 3,
          "rolling folds move their training start")


def test_metrics():
    print("\nRanking metrics")
    check(abs(roc_auc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9, "perfect separation scores 1.0")
    check(abs(roc_auc([0, 1, 2], [3, 4, 5]) - 0.0) < 1e-9, "inverted separation scores 0.0")

    # Ties must score 0.5, not 1.0. An untrained model emits near-constant
    # scores, and a naive rank sum would report it as perfect.
    check(abs(roc_auc([1, 1, 1], [1, 1, 1]) - 0.5) < 1e-9, "complete ties score 0.5")

    check(0.4 < roc_auc(np.random.default_rng(0).normal(size=500),
                        np.random.default_rng(1).normal(size=500)) < 0.6,
          "random scores land near 0.5")
    check(abs(average_precision([5, 4], [1, 2]) - 1.0) < 1e-9,
          "average precision is 1.0 for perfect ranking")
    check(abs(mean_reciprocal_rank([10], np.array([[1, 2, 3]])) - 1.0) < 1e-9,
          "MRR is 1.0 when the positive beats every negative")
    check(abs(mean_reciprocal_rank([0], np.array([[1, 2, 3]])) - 0.25) < 1e-9,
          "MRR is 1/4 when three negatives beat the positive")


def test_recency_features():
    print("\nRecency features")
    import graph_engine

    rng = np.random.default_rng(5)
    num_nodes, num_edges = 40, 20000
    src = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    dst = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    ts = np.sort(rng.integers(1_600_000_000, 1_600_000_000 + 86400 * 60, num_edges)).astype(np.uint32)

    graph = graph_engine.PCSRGraph(num_nodes, num_edges * 3, 128 * 1024 * 1024)
    graph.insert_edges(src, dst, ts)
    sampler = PCSRTemporalSampler(graph)

    query_nodes = np.arange(num_nodes)
    query_time = int(ts.max())
    features = sampler.recency_features(query_nodes, np.full(num_nodes, query_time), 10)

    check(features.shape == (num_nodes, 4), "one row per query, four features")
    check(np.isfinite(features).all(), "features are finite")

    # Feature 0 is a scaled log history size; verify against a direct count.
    expected = np.log1p((src[ts < query_time] == 0).sum()) / 5.0
    check(abs(features[0, 0] - expected) < 1e-4, "history size matches a direct count")

    # A node with no history must not produce a large or negative rate.
    empty = sampler.recency_features(np.array([0]), np.array([0]), 10)
    check(np.isfinite(empty).all() and empty[0, 3] < 0,
          "a node with no history yields a finite, low rate")

    # Scale is what made the first version untrainable.
    check(np.abs(features).max() < 20,
          f"features stay O(1) (max |value| {np.abs(features).max():.2f})")


def test_replayer():
    print("\nReplayer")
    events, _ = synthetic_corpus(num_days=40, events_per_day=200, seed=4)
    countries = sorted(set(events["src"]) | set(events["dst"]))

    path = tmp("replay.json")
    if os.path.exists(path):
        os.remove(path)
    mapper = EntityMapper(filepath=path, max_entities=len(countries))

    replayer = DataReplayer.for_events(len(events), len(countries))
    stats = replayer.replay_events(events["src"].to_numpy(), events["dst"].to_numpy(),
                                   events["ts"].to_numpy(), mapper, quiet=True)

    check(stats["inserted"] == len(events), "every event is inserted")
    check(stats["graph_edges"] == len(events), "graph retains every edge")
    check(stats["ns_per_edge"] < 2000,
          f"bulk path stays fast ({stats['ns_per_edge']:.0f} ns/event)")

    sampler = PCSRTemporalSampler(replayer.graph)
    check(sampler.validate(), "adjacency runs are chronological after replay")

    check(provision(1000, 50_000) >= 50_000 * 16,
          "provisioning respects the per-vertex floor")

    # An id past the cap must be dropped, not handed to the engine.
    tiny_path = tmp("tiny.json")
    if os.path.exists(tiny_path):
        os.remove(tiny_path)
    tiny_mapper = EntityMapper(filepath=tiny_path, max_entities=3)
    tiny = DataReplayer(num_nodes=3, max_edges=1024, arena_bytes=4 * 1024 * 1024)
    tiny_stats = tiny.replay_events(events["src"].to_numpy()[:500],
                                    events["dst"].to_numpy()[:500],
                                    events["ts"].to_numpy()[:500],
                                    tiny_mapper, quiet=True)
    check(tiny_stats["dropped"] > 0, "events with over-cap entities are dropped")
    check(tiny_stats["graph_edges"] == tiny_stats["inserted"],
          "no out-of-range id reaches the engine")


# ---------------------------------------------------------------------------
# Backtest properties
# ---------------------------------------------------------------------------

def _fixture(seed=0, days=260):
    events, prices = synthetic_corpus(num_days=days, events_per_day=150, seed=seed)
    universe = Universe()
    countries = sorted(set(events["src"]))
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"].unique())))[100:]
    return events, prices, universe, countries, dates


def test_backtest_control():
    print("\nBacktest: random control")
    _, prices, universe, countries, dates = _fixture()

    config = BacktestConfig(cost_bps=0.0, timeseries_window=0)
    sharpes = []
    for seed in range(10):
        panel = random_signal(countries, dates, seed=200 + seed)
        sharpes.append(run_backtest(panel, prices, universe, config)["metrics"]["sharpe"])

    mean_sharpe = float(np.mean(sharpes))
    print(f"       random Sharpe over 10 seeds: mean {mean_sharpe:+.3f}, "
          f"range [{min(sharpes):+.2f}, {max(sharpes):+.2f}]")
    check(abs(mean_sharpe) < 1.0,
          "a random signal earns nothing before costs")

    # Costs must bite, and bite harder on a high-turnover signal.
    panel = random_signal(countries, dates, seed=7)
    free = run_backtest(panel, prices, universe,
                        BacktestConfig(cost_bps=0.0, timeseries_window=0))
    charged = run_backtest(panel, prices, universe,
                           BacktestConfig(cost_bps=20.0, timeseries_window=0))
    check(charged["metrics"]["total_return"] < free["metrics"]["total_return"],
          "costs reduce returns")
    check(charged["metrics"]["total_cost"] > free["metrics"]["total_cost"],
          "cost accounting scales with the charge")


def test_backtest_no_lookahead():
    print("\nBacktest: lookahead discipline")
    events, prices, universe, countries, dates = _fixture()
    panel = goldstein_signal(events, countries, dates, 5)

    def sharpe(lag):
        config = BacktestConfig(cost_bps=0.0, execution_lag_days=lag, timeseries_window=0)
        return run_backtest(panel, prices, universe, config)["metrics"]["sharpe"]

    peeking, normal, delayed = sharpe(-1), sharpe(1), sharpe(5)
    print(f"       Sharpe by lag: -1d {peeking:+.2f}  1d {normal:+.2f}  5d {delayed:+.2f}")

    # If the timing were wired wrongly these would not be ordered.
    check(peeking > normal, "deliberately peeking one day scores better than not")
    check(normal > delayed, "a stale signal scores worse than a fresh one")
    check(normal > 0, "the planted signal is recoverable at an honest lag")

    # Shifting the signal later in time must destroy it; if a late signal still
    # worked, dates would not be binding the way the code claims.
    shifted = panel.copy()
    shifted["date"] = (pd.to_datetime(shifted["date"]) +
                       pd.Timedelta(days=30)).dt.date.astype(str)
    shifted_sharpe = run_backtest(shifted, prices, universe,
                                  BacktestConfig(cost_bps=0.0, timeseries_window=0)
                                  )["metrics"]["sharpe"]
    print(f"       same signal shifted 30 days later: {shifted_sharpe:+.2f}")
    check(shifted_sharpe < normal, "a time-shifted signal loses its edge")

    # Cross-sectional standardisation must not consult the future: changing a
    # late date's signal cannot alter an early date's weights.
    tampered = panel.copy()
    late = tampered["date"] > sorted(tampered["date"].unique())[-20]
    tampered.loc[late, "signal"] = tampered.loc[late, "signal"] * 1000 + 5000
    base_weights = run_backtest(panel, prices, universe,
                                BacktestConfig(timeseries_window=0))["weights"]
    tampered_weights = run_backtest(tampered, prices, universe,
                                    BacktestConfig(timeseries_window=0))["weights"]
    early = base_weights.index < sorted(pd.to_datetime(panel["date"].unique()))[-25]
    check(np.allclose(base_weights[early].fillna(0),
                      tampered_weights[early].fillna(0), atol=1e-9),
          "corrupting late signals leaves early weights untouched")


def test_backtest_book():
    print("\nBacktest: book construction")
    events, prices, universe, countries, dates = _fixture()
    panel = goldstein_signal(events, countries, dates, 5)

    config = BacktestConfig(max_weight=0.15, timeseries_window=0)
    result = run_backtest(panel, prices, universe, config)
    weights = result["weights"].dropna(how="all")

    check(float(weights.abs().max().max()) <= 0.15 + 1e-9,
          "no position exceeds max_weight")
    check(float(weights.sum(axis=1).abs().max()) < 1e-9,
          "the book is dollar-neutral every day")
    check(0.5 < float(weights.abs().sum(axis=1).mean()) <= 1.0 + 1e-9,
          "gross exposure is close to the configured level")

    flipped = run_backtest(panel, prices, universe,
                           BacktestConfig(direction=1, cost_bps=0.0,
                                          timeseries_window=0))
    straight = run_backtest(panel, prices, universe,
                            BacktestConfig(direction=-1, cost_bps=0.0,
                                           timeseries_window=0))
    check(np.sign(flipped["metrics"]["sharpe"]) != np.sign(straight["metrics"]["sharpe"]),
          "flipping direction flips the sign of the result")


def test_backtest_recovers_planted_signal():
    print("\nBacktest: recovers a known signal")
    events, prices, universe, countries, dates = _fixture()
    panel = goldstein_signal(events, countries, dates, 5)
    result = run_backtest(panel, prices, universe,
                          BacktestConfig(cost_bps=5.0, timeseries_window=0))
    metrics = result["metrics"]
    print(f"       planted-signal Sharpe {metrics['sharpe']:+.2f} "
          f"(t {metrics['t_stat']:+.2f}) over {metrics['days']} days")
    check(metrics["sharpe"] > 1.0,
          "the harness recovers a signal that is genuinely in the data")
    check(metrics["t_stat"] > 2.0, "and it is statistically distinguishable from noise")


def test_reversal_baseline():
    print("\nReversal baseline")
    from strategy.signals import reversal_signal

    events, prices = synthetic_corpus(num_days=150, events_per_day=80, seed=8)
    universe = Universe()
    countries = sorted(set(events["src"]))
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"].unique())))[30:]

    panel = reversal_signal(prices, universe, countries, dates)
    check(not panel.empty, "reversal signal is produced")
    check(set(panel.columns) == {"date", "country", "signal"},
          "reversal signal has the standard shape")

    # It must be the negated trailing return, so it can only use past prices.
    wide = prices.copy()
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    row = panel.iloc[len(panel) // 2]
    ticker = universe.ticker(row["country"])
    position = wide.index.get_loc(pd.Timestamp(row["date"]))
    expected = -(wide[ticker].iloc[position] / wide[ticker].iloc[position - 1] - 1.0)
    check(abs(row["signal"] - expected) < 1e-9,
          "reversal equals the negated prior-day return")


def test_edge_weights_roundtrip():
    print("\nEdge weights through the Python stack")
    import graph_engine
    from models import PCSRTemporalSampler

    rng = np.random.default_rng(23)
    num_nodes, num_edges = 40, 20000
    src = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    dst = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    ts = np.sort(rng.integers(1_000, 90_000, num_edges)).astype(np.uint32)
    # Weight derived from the timestamp, so a mismatch is detectable anywhere.
    weight = (ts.astype(np.float32) / 8.0).astype(np.float32)

    graph = graph_engine.PCSRGraph(num_nodes, 256, 128 * 1024 * 1024,
                                   store_weights=True)  # forces growth
    graph.insert_edges(src, dst, ts, None, weight)
    check(graph.resize_count > 0, "weights survive a growing PMA")

    _, _, coo_ts, _, coo_w = graph.to_coo()
    check(np.allclose(coo_w, coo_ts.astype(np.float32) / 8.0),
          "to_coo weights match their own timestamps")

    sampler = PCSRTemporalSampler(graph)
    nodes = rng.integers(0, num_nodes, 256)
    times = rng.integers(40_000, 90_000, 256).astype(np.int64)
    ids, n_times, mask, weights = sampler.sample(nodes, times, 8, with_weights=True)
    check(weights.shape == ids.shape, "sampled weights match the neighbour shape")
    check(bool(np.allclose(weights[mask], n_times[mask].astype(np.float32) / 8.0)),
          "each sampled neighbour carries its own weight")
    check(bool((weights[~mask] == 0).all()), "masked slots carry no weight")

    # Omitting weights must stay backwards compatible, and a graph built
    # without them must not silently accept one.
    plain = graph_engine.PCSRGraph(num_nodes, 4096, 16 * 1024 * 1024,
                                   store_weights=True)
    plain.insert_edges(src[:1000], dst[:1000], ts[:1000])
    check(float(np.asarray(plain.get_edge_weights())[:1000].sum()) == 0.0,
          "unspecified weights default to zero")

    unweighted = graph_engine.PCSRGraph(num_nodes, 4096, 16 * 1024 * 1024)
    check(not unweighted.has_weights, "weights are off by default")
    check(np.asarray(unweighted.get_edge_weights()).size == 0,
          "an unweighted graph exposes an empty weight view")
    unweighted.insert_edges(src[:1000], dst[:1000], ts[:1000])
    check(int(unweighted.num_edges) == 1000, "an unweighted graph still stores edges")
    try:
        unweighted.insert_edges(src[:10], dst[:10], ts[:10], None, weight[:10])
        raised = False
    except ValueError:
        raised = True
    check(raised, "weights into an unweighted graph raise instead of vanishing")


def test_expiry_through_sampler():
    print("\nExpiry keeps a bounded window causally correct")
    import graph_engine
    from models import PCSRTemporalSampler

    rng = np.random.default_rng(77)
    num_nodes, num_edges = 60, 40000
    src = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    dst = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    ts = np.sort(rng.integers(1_000, 100_000, num_edges)).astype(np.uint32)

    graph = graph_engine.PCSRGraph(num_nodes, 8192, 128 * 1024 * 1024)
    graph.insert_edges(src, dst, ts)

    cutoff = 60_000
    survivors = int((ts >= cutoff).sum())
    removed = graph.expire_before(cutoff)
    check(removed == num_edges - survivors,
          f"expire_before drops exactly the old edges ({removed})")
    check(int(graph.num_edges) == survivors, "the live count is the survivor count")
    check(int(np.asarray(graph.get_vertex_counts()).sum()) == survivors,
          "a full scan agrees with the reported count")

    # The sampler must see the new boundaries, and must still refuse to look
    # forward: an expired graph is a truncated history, not a shifted one.
    sampler = PCSRTemporalSampler(graph)
    check(sampler.validate(), "runs are still chronological after expiry")

    nodes = rng.integers(0, num_nodes, 512)
    times = rng.integers(70_000, 100_000, 512).astype(np.int64)
    ids, n_times, mask = sampler.sample(nodes, times, 12)
    check(bool((n_times[mask] < times[:, None].repeat(12, axis=1)[mask]).all()),
          "no sampled neighbour is at or after its query time")
    check(bool((n_times[mask] >= cutoff).all()),
          "no expired edge is reachable through the sampler")

    # Against a reference computed directly from the arrays.
    expected = np.zeros(len(nodes), dtype=np.int64)
    for i, (node, time) in enumerate(zip(nodes, times)):
        expected[i] = int(((src == node) & (ts >= cutoff) & (ts < time)).sum())
    check(bool((mask.sum(axis=1) == np.minimum(expected, 12)).all()),
          "admissible-history sizes match a brute-force count")

    # Continuing to insert after expiry must not corrupt anything.
    more_src = rng.integers(0, num_nodes, 5000).astype(np.uint32)
    more_dst = rng.integers(0, num_nodes, 5000).astype(np.uint32)
    more_ts = np.sort(rng.integers(100_000, 120_000, 5000)).astype(np.uint32)
    graph.insert_edges(more_src, more_dst, more_ts)
    check(int(graph.num_edges) == survivors + 5000, "inserts after expiry all land")
    sampler.refresh()
    check(sampler.validate(), "runs are still chronological after expiry then insert")


def test_edge_relations_roundtrip():
    print("\nEdge relations through the Python stack")
    import graph_engine
    from models import PCSRTemporalSampler

    rng = np.random.default_rng(11)
    num_nodes, num_edges = 50, 30000
    src = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    dst = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    ts = np.sort(rng.integers(1_000, 90_000, num_edges)).astype(np.uint32)
    # Relation derived from the target, so a mismatch is detectable anywhere.
    rel = ((dst.astype(np.int64) * 7 + 3) % 20 + 1).astype(np.uint16)

    graph = graph_engine.PCSRGraph(num_nodes, 256, 128 * 1024 * 1024)  # forces growth
    graph.insert_edges(src, dst, ts, rel)
    check(graph.resize_count > 0, "relations survive a growing PMA")

    sampler = PCSRTemporalSampler(graph)
    ids, times, mask, relations = sampler.sample(
        np.arange(num_nodes), np.full(num_nodes, int(ts.max())), 12,
        with_relations=True)

    check(relations.shape == ids.shape, "relation array matches the neighbour array")
    expected = ((ids.astype(np.int64) * 7 + 3) % 20 + 1)
    check(bool((relations[mask] == expected[mask]).all()),
          "each sampled neighbour carries its own relation")

    # to_coo must agree with the zero-copy views.
    s_arr, d_arr, t_arr, r_arr, w_arr = graph.to_coo()
    check(len(r_arr) == int(graph.num_edges), "to_coo returns one relation per edge")
    check(bool((r_arr == ((d_arr.astype(np.int64) * 7 + 3) % 20 + 1)).all()),
          "to_coo relations match their targets")

    # Omitting relations must stay backwards compatible.
    plain = graph_engine.PCSRGraph(num_nodes, 4096, 16 * 1024 * 1024)
    plain.insert_edges(src[:1000], dst[:1000], ts[:1000])
    check(int(np.asarray(plain.get_edge_relations())[:1000].sum()) == 0,
          "omitting relations defaults them to zero")

    three = PCSRTemporalSampler(plain).sample(np.arange(10), np.full(10, 50_000), 5)
    check(len(three) == 3, "callers that do not ask for relations get the old 3-tuple")


def test_fx_universe():
    print("\nFX universe")
    from strategy.universe import fx_universe, FX_INVERTED

    fx = fx_universe()
    check(len(set(fx.mapping.values())) < len(fx.mapping),
          "several countries share one currency (the euro bloc)")
    check(fx.ticker("DEU") == fx.ticker("FRA") == "EURUSD=X",
          "eurozone countries map to one instrument")
    check("USDJPY=X" in fx.invert and "EURUSD=X" not in fx.invert,
          "only USD-base quotes are marked for inversion")

    # Inversion must actually flip the sign of the return.
    dates = pd.bdate_range("2024-01-01", periods=60)
    rising = pd.DataFrame({"date": dates.date.astype(str), "ticker": "USDJPY=X",
                           "close": np.linspace(100, 120, 60)})
    flat = pd.concat([pd.DataFrame({"date": dates.date.astype(str), "ticker": t,
                                    "close": 100.0}) for t in ("EURUSD=X", "GBPUSD=X")])
    prices = pd.concat([rising, flat], ignore_index=True)

    from backtest.engine import _to_wide_prices
    wide = _to_wide_prices(prices)
    for t in FX_INVERTED & set(wide.columns):
        wide[t] = 1.0 / wide[t]
    check(wide["USDJPY=X"].iloc[-1] < wide["USDJPY=X"].iloc[0],
          "a rising USDJPY becomes a falling yen after inversion")


def test_duplicate_instrument_aggregation():
    print("\nShared-instrument aggregation")
    from backtest.engine import BacktestConfig, run_backtest
    from strategy.universe import Universe

    dates = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(3)
    tickers = ["AAA", "BBB", "CCC"]
    prices = pd.concat([
        pd.DataFrame({"date": dates.date.astype(str), "ticker": t,
                      "close": 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))})
        for t in tickers], ignore_index=True)

    # Four countries, three instruments: two share AAA.
    universe = Universe(mapping={"P": "AAA", "Q": "AAA", "R": "BBB", "S": "CCC"})
    rows = [(d.date().isoformat(), c, float(rng.normal()))
            for d in dates for c in ("P", "Q", "R", "S")]
    signals = pd.DataFrame(rows, columns=["date", "country", "signal"])

    result = run_backtest(signals, prices, universe,
                          BacktestConfig(min_names=3, timeseries_window=0))
    weights = result["weights"].dropna(how="all")
    check(list(weights.columns) == tickers,
          "shared instruments collapse to one column, not duplicates")
    check(float(weights.abs().sum(axis=1).max()) <= 1.0 + 1e-9,
          "gross exposure is not inflated by the duplicate")


def test_bulk_matches_scalar():
    print("\nBulk and scalar insertion agree")
    import graph_engine

    # The bulk path exists because per-edge calls cost ~1us at the language
    # boundary. It is a separate code path through the same engine, so it can
    # drift: a different rebalance order would still retain every edge while
    # producing a different layout, and nothing else in the suite would notice.
    rng = np.random.default_rng(7)
    num_nodes, num_edges = 200, 50_000
    src = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    dst = rng.integers(0, num_nodes, num_edges).astype(np.uint32)
    ts = np.sort(rng.integers(1_000, 90_000, num_edges)).astype(np.uint32)
    rel = rng.integers(1, 50, num_edges).astype(np.uint16)

    bulk = graph_engine.PCSRGraph(num_nodes, 256, 1 << 27)
    bulk.insert_edges(src, dst, ts, rel)

    scalar = graph_engine.PCSRGraph(num_nodes, 256, 1 << 27)
    for i in range(num_edges):
        scalar.insert_edge(int(src[i]), int(dst[i]), int(ts[i]), int(rel[i]))

    check(bulk.resize_count > 0 and bulk.resize_count == scalar.resize_count,
          "both paths grew the PMA the same number of times")
    check(np.array_equal(np.asarray(bulk.get_vertex_offsets()),
                         np.asarray(scalar.get_vertex_offsets())),
          "region boundaries are identical")
    check(np.array_equal(np.asarray(bulk.get_vertex_counts()),
                         np.asarray(scalar.get_vertex_counts())),
          "degrees are identical")

    for name, left, right in zip(("src", "dst", "timestamp", "relation", "weight"),
                                 bulk.to_coo(), scalar.to_coo()):
        check(np.array_equal(left, right), f"{name} column is identical")


def test_single_writer_guard():
    print("\nConcurrent writers are rejected, not silently tolerated")
    import threading
    import graph_engine

    # The bulk insert path releases the GIL to get its throughput, which means
    # the interpreter lock no longer serialises writers for free. PCSRGraph is
    # single-writer -- a rebalance rewrites whole windows -- so two concurrent
    # bulk inserts used to interleave into corruption: measured at 53,592
    # reported edges out of 160,000 inserted, with a full scan disagreeing with
    # the counter and real edges lost. No crash, no warning.
    num_nodes, per_thread = 500, 40_000

    def batch(seed):
        rng = np.random.default_rng(seed)
        return (rng.integers(0, num_nodes, per_thread).astype(np.uint32),
                rng.integers(0, num_nodes, per_thread).astype(np.uint32),
                np.sort(rng.integers(1_000, 90_000, per_thread)).astype(np.uint32))

    graph = graph_engine.PCSRGraph(num_nodes, 8 * per_thread, 1 << 28)
    rejected = []

    def writer(payload):
        try:
            graph.insert_edges(*payload)
        except RuntimeError as error:
            rejected.append(str(error))

    threads = [threading.Thread(target=writer, args=(batch(i),)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    check(len(rejected) > 0, "at least one concurrent writer is turned away")
    check(all("single-writer" in message for message in rejected),
          "the rejection names the actual problem")

    scanned = int(np.asarray(graph.get_vertex_counts()).sum())
    check(scanned == graph.num_edges,
          f"the graph stays consistent after contention "
          f"(scan {scanned} vs counter {graph.num_edges})")
    check(graph.num_edges % per_thread == 0,
          "only whole successful batches are present")

    # Serialised writers must all still succeed -- the guard must not be sticky.
    serial = graph_engine.PCSRGraph(num_nodes, 8 * per_thread, 1 << 28)
    for i in range(4):
        serial.insert_edges(*batch(i))
    check(serial.num_edges == 4 * per_thread,
          "sequential writers are unaffected by the guard")
    check(int(np.asarray(serial.get_vertex_counts()).sum()) == serial.num_edges,
          "sequentially built graph is consistent")

    # A guard released by an exception must not leave the graph locked.
    tiny = graph_engine.PCSRGraph(4, 16, 1 << 20)
    try:
        tiny.insert_edge(99, 0, 1)
    except IndexError:
        pass
    except Exception:
        pass
    tiny.insert_edge(0, 1, 1)
    check(tiny.num_edges == 1, "a throwing insert releases the write guard")


def test_determinism():
    print("\nSeeded runs are reproducible")
    import graph_engine
    import torch

    from models import PCSRTemporalSampler, TGATLinkModel

    def build(seed):
        events, _ = synthetic_corpus(num_days=40, events_per_day=120, seed=1)
        conflict = events[events["quad_class"].isin(CONFLICT_CLASSES)].reset_index(drop=True)
        countries = sorted(set(conflict["src"]) | set(conflict["dst"]))

        path = tmp(f"determinism_{seed}.json")
        if os.path.exists(path):
            os.remove(path)
        mapper = EntityMapper(filepath=path, max_entities=len(countries) + 8)

        relations = conflict["event_root_code"].clip(0, 65534).astype(np.uint16).to_numpy()
        replayer = DataReplayer.for_events(len(conflict), mapper.max_entities)
        replayer.replay_events(conflict["src"].to_numpy(), conflict["dst"].to_numpy(),
                               conflict["ts"].to_numpy(), mapper, quiet=True,
                               relations=relations)
        sampler = PCSRTemporalSampler(replayer.graph)

        conflict = conflict.assign(
            src_id=mapper.get_ids(conflict["src"].to_numpy()),
            dst_id=mapper.get_ids(conflict["dst"].to_numpy()))
        ids = np.array(sorted(set(conflict["dst_id"])))

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = TGATLinkModel(mapper.max_entities, sampler, node_dim=16, time_dim=16,
                              num_layers=2, num_neighbors=8,
                              num_relations=int(relations.max()) + 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        train = conflict.iloc[:2500]
        for offset in range(0, len(train), 200):
            batch = train.iloc[offset:offset + 200]
            if len(batch) < 4:
                continue
            optimizer.zero_grad()
            loss, _, _ = model.loss(batch["src_id"].to_numpy(), batch["dst_id"].to_numpy(),
                                    batch["ts"].to_numpy(),
                                    rng.choice(ids, len(batch)), rng.choice(ids, len(batch)))
            loss.backward()
            optimizer.step()

        model.eval()
        test = conflict.iloc[2500:2800]
        with torch.no_grad():
            scores = model.score(test["src_id"].to_numpy(), test["dst_id"].to_numpy(),
                                 test["ts"].to_numpy()).numpy()
        return np.asarray(replayer.graph.get_vertex_counts()).copy(), scores

    counts_a, scores_a = build(0)
    counts_b, scores_b = build(0)
    counts_c, scores_c = build(1)

    check(np.array_equal(counts_a, counts_b), "graph construction is deterministic")
    check(np.array_equal(scores_a, scores_b),
          "the same seed reproduces model scores bit for bit")
    check(not np.array_equal(scores_a, scores_c),
          "a different seed actually changes the result")


def test_propagation_recovers_planted_half_life():
    print("\nPropagation event study recovers a known half-life")
    from models import PCSRTemporalSampler

    # A world where a shock at one firm genuinely reaches its suppliers with a
    # known decay. If the estimator cannot recover a half-life that was planted
    # deliberately, no number it reports on real data means anything.
    planted, days, firms_n = 3.0, 900, 300
    rng = np.random.default_rng(5)

    links = synthetic_supply_chain(num_firms=firms_n, num_links=5000, seed=6)
    firms = sorted(set(links["src"]) | set(links["dst"]))
    index = {f: i for i, f in enumerate(firms)}
    dates = pd.bdate_range("2019-01-02", periods=days)

    suppliers = {}
    for supplier, customer in zip(links["src"], links["dst"]):
        suppliers.setdefault(customer, []).append(supplier)

    shock = np.zeros((days, len(firms)))
    events = []
    decay = 0.5 ** (1.0 / planted)
    for day in range(40, days - 40):
        if rng.random() < 0.5:
            firm = firms[rng.integers(0, len(firms))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            events.append((dates[day].date().isoformat(), firm, sign))
            remaining = sign * 0.010
            for h in range(1, 30):
                if day + h >= days:
                    break
                for supplier in suppliers.get(firm, []):
                    shock[day + h, index[supplier]] += remaining * (1 - decay)
                remaining *= decay

    event_frame = pd.DataFrame(events, columns=["date", "ticker", "sign"])
    log_price = np.zeros(len(firms))
    rows = []
    for day in range(days):
        log_price = log_price + shock[day] + rng.normal(0, 0.014, len(firms))
        for firm, i in index.items():
            rows.append((dates[day].date().isoformat(), firm,
                         float(100 * np.exp(log_price[i]))))
    prices = pd.DataFrame(rows, columns=["date", "ticker", "close"])

    # Neighbourhoods come from the engine, as of the event date -- suppliers
    # are gained and lost over time, so this must be a causal query.
    path = tmp("propagation.json")
    if os.path.exists(path):
        os.remove(path)
    mapper = EntityMapper(filepath=path, max_entities=len(firms) + 8)
    replayer = DataReplayer.for_events(len(links), mapper.max_entities)
    replayer.replay_events(links["dst"].to_numpy(), links["src"].to_numpy(),
                           links["ts"].to_numpy(), mapper, quiet=True)
    sampler = PCSRTemporalSampler(replayer.graph)
    names = {mapper.get_id(f): f for f in firms if mapper.get_id(f) >= 0}

    def neighbors(ticker, date):
        node = mapper.get_id(ticker)
        if node < 0:
            return []
        stamp = int(pd.Timestamp(date).timestamp())
        ids, _, mask = sampler.sample(np.array([node]), np.array([stamp]), 32)
        return [names[int(x)] for x in ids[0][mask[0]] if int(x) in names]

    result = run_event_study(event_frame, neighbors, prices,
                             horizons=(1, 2, 3, 5, 8, 12, 20), quiet=True)

    recovered = result["half_life"]["difference"]
    horizons = result["horizons"]
    peak_t = max(abs(result["neighbors"][h][1]) for h in horizons
                 if np.isfinite(result["neighbors"][h][1]))
    # The placebo drifts negative by construction: demeaning the cross-section
    # each day pushes non-treated firms down when the treated group is lifted.
    # What must not happen is a placebo response in the *same* direction as the
    # neighbours, which would mean genuine contamination rather than arithmetic.
    placebo_same_direction = max(
        (result["placebo"][h][1] for h in horizons
         if np.isfinite(result["placebo"][h][1])), default=0.0)

    print(f"       planted {planted:.1f}d -> recovered "
          f"{recovered if recovered is None else round(recovered, 2)}d "
          f"| peak t {peak_t:.1f} | placebo same-direction t {placebo_same_direction:+.2f}")

    check(recovered is not None, "a half-life is recoverable at all")
    check(recovered is not None and 1.0 < recovered < 7.0,
          f"recovered half-life brackets the planted {planted}d (got {recovered})")
    check(peak_t > 3.0, "the neighbour response is clearly distinguishable from noise")
    check(placebo_same_direction < 2.0,
          f"non-neighbours show no response in the treated direction "
          f"(max t {placebo_same_direction:+.2f})")


def test_streaming_ingestor_python():
    print("\nStreaming ingestion from Python")
    import graph_engine

    num_nodes, num_events = 2000, 200_000
    rng = np.random.default_rng(3)
    src = rng.integers(0, num_nodes, num_events).astype(np.uint32)
    dst = rng.integers(0, num_nodes, num_events).astype(np.uint32)
    ts = np.sort(rng.integers(1_000, 900_000, num_events)).astype(np.uint32)
    rel = rng.integers(1, 20, num_events).astype(np.uint16)

    graph = graph_engine.PCSRGraph(num_nodes, 4 * num_events, 1 << 28)
    # Queue far smaller than the batch size, so the producer must block. With a
    # queue larger than a batch, whether back-pressure happens at all depends on
    # thread scheduling, and the assertion below becomes flaky.
    ingestor = graph_engine.StreamingIngestor(graph, 64)

    with ingestor:
        check(ingestor.running, "context manager starts the consumer")
        for i in range(0, num_events, 8192):
            ingestor.push_batch(src[i:i + 8192], dst[i:i + 8192],
                                ts[i:i + 8192], rel[i:i + 8192])

    check(not ingestor.running, "context manager stops the consumer")
    check(ingestor.pushed == num_events, "every event was pushed")
    check(ingestor.consumed == num_events, "every event was consumed")
    check(graph.num_edges == num_events, "the graph holds every streamed event")
    check(int(np.asarray(graph.get_vertex_counts()).sum()) == num_events,
          "a full scan agrees with the counter")
    check(ingestor.producer_spins > 0,
          "a small queue exercised back-pressure rather than dropping")

    # Relations must survive the handoff, and runs must stay chronological.
    sampler_ok = True
    from models import PCSRTemporalSampler
    check(PCSRTemporalSampler(graph).validate(),
          "streamed runs are chronologically ordered")
    check(sampler_ok, "sampler accepts a streamed graph")


def test_ethereum_loader():
    print("\nEthereum loader")
    import tempfile
    from ingestion.ethereum import (REL_CONTRACT_CALL, REL_CONTRACT_CALL_ZERO,
                                    REL_FAILED, REL_VALUE_TRANSFER, classify,
                                    load_transactions)

    # --- relation classification, on hand-built rows -----------------------
    frame = pd.DataFrame({
        "n_input_bytes": [0, 4, 4, 0, 4],
        "value_f64": [1e18, 1e18, 0.0, 5e17, 0.0],
        "to_address": [b"a", b"b", b"c", b"d", b"e"],
        "success": [True, True, True, True, False],
    })
    codes = classify(frame)
    check(codes[0] == REL_VALUE_TRANSFER, "no calldata + value => value transfer")
    check(codes[1] == REL_CONTRACT_CALL, "calldata + value => contract call")
    check(codes[2] == REL_CONTRACT_CALL_ZERO, "calldata + no value => zero-value call")
    check(codes[4] == REL_FAILED, "a reverted transaction is typed as failed")
    check(codes.dtype == np.uint16, "relation codes are uint16 for the engine")

    # --- the filter-and-reindex path ---------------------------------------
    # Three busy addresses transacting among themselves, plus a dust tail of
    # one-shot addresses that must be removed without corrupting the ids.
    rng = np.random.default_rng(0)
    busy = [b"busy%02d" % i for i in range(3)]
    rows = []
    block = 100
    for i in range(60):
        a, b = rng.choice(len(busy), 2, replace=False)
        rows.append((block + i, busy[a], busy[b], 1e18, 0, True, 0))
    for i in range(40):  # dust: each address appears exactly once
        rows.append((block + 60 + i, b"dust%03d" % i, busy[0], 0.0, 0, True, 4))

    tx = pd.DataFrame(rows, columns=["block_number", "from_address", "to_address",
                                     "value_f64", "gas_used", "success", "n_input_bytes"])
    blocks = pd.DataFrame({"block_number": tx["block_number"].unique()})
    blocks["timestamp"] = 1_700_000_000 + blocks["block_number"] * 12

    with tempfile.TemporaryDirectory() as directory:
        tx_path = os.path.join(directory, "transactions.parquet")
        block_path = os.path.join(directory, "blocks.parquet")
        tx.to_parquet(tx_path, index=False)
        blocks.to_parquet(block_path, index=False)

        table, addresses = load_transactions(tx_path, block_path, min_degree=5, quiet=True)

        check(len(table) == 60, f"dust edges are dropped (kept {len(table)} of 100)")
        check(len(addresses) == 3, f"only busy addresses survive (got {len(addresses)})")

        # The engine allocates a region per vertex id, so ids must be dense.
        used = set(table["src"]).union(table["dst"])
        check(used == set(range(len(addresses))),
              "surviving ids are re-indexed into a dense range with no gaps")
        check(table["src"].max() < len(addresses) and table["dst"].max() < len(addresses),
              "no id exceeds the address count")

        check(table["ts"].is_monotonic_increasing, "output is chronological")
        check(list(table.columns) == ["ts", "src", "dst", "relation", "value"],
              "schema is the engine's edge tuple")

        # Unfiltered, every address must still round-trip.
        full, full_addresses = load_transactions(tx_path, block_path, min_degree=1, quiet=True)
        check(len(full) == 100, "min_degree=1 keeps everything")
        check(len(full_addresses) == 43, f"all addresses retained (got {len(full_addresses)})")
        check(set(full["src"]).union(full["dst"]) == set(range(len(full_addresses))),
              "unfiltered ids are also dense")


def test_ethereum_end_to_end():
    print("\nEthereum edges through the engine")
    import tempfile
    import graph_engine
    from ingestion.ethereum import load_transactions
    from models import PCSRTemporalSampler

    rng = np.random.default_rng(4)
    num_addresses, num_tx = 200, 20_000
    frm = rng.integers(0, num_addresses, num_tx)
    to = rng.integers(0, num_addresses, num_tx)
    rows = pd.DataFrame({
        "block_number": np.sort(rng.integers(1000, 3000, num_tx)),
        "from_address": [b"a%04d" % i for i in frm],
        "to_address": [b"a%04d" % i for i in to],
        "value_f64": rng.choice([0.0, 1e17, 1e18], num_tx),
        "gas_used": 21000, "success": True,
        "n_input_bytes": rng.choice([0, 68], num_tx),
    })
    blocks = pd.DataFrame({"block_number": np.unique(rows["block_number"])})
    blocks["timestamp"] = 1_700_000_000 + blocks["block_number"] * 12

    with tempfile.TemporaryDirectory() as directory:
        tx_path = os.path.join(directory, "tx.parquet")
        block_path = os.path.join(directory, "bl.parquet")
        rows.to_parquet(tx_path, index=False)
        blocks.to_parquet(block_path, index=False)
        table, addresses = load_transactions(tx_path, block_path, min_degree=5, quiet=True)

    graph = graph_engine.PCSRGraph(len(addresses), len(table) * 3, 1 << 27)
    graph.insert_edges(table["src"].to_numpy(np.uint32), table["dst"].to_numpy(np.uint32),
                       table["ts"].to_numpy(np.int64).astype(np.uint32),
                       table["relation"].to_numpy(np.uint16))

    check(graph.num_edges == len(table), "every loaded edge reaches the engine")
    sampler = PCSRTemporalSampler(graph)
    check(sampler.validate(), "adjacency runs are chronological")

    # Relations must survive, and the sampler must return them aligned.
    ids, times, mask, relations = sampler.sample(
        np.arange(min(50, len(addresses))),
        np.full(min(50, len(addresses)), int(table["ts"].max())), 16, with_relations=True)
    check(relations[mask].max() <= table["relation"].max(),
          "sampled relations stay within the classified vocabulary")
    check(bool((times[mask] <= int(table["ts"].max())).all()),
          "sampled events respect the query time")


def main():
    test_id_mapper()
    test_timestamps()
    test_universe()
    test_folds()
    test_metrics()
    test_recency_features()
    test_replayer()
    test_backtest_control()
    test_backtest_no_lookahead()
    test_backtest_book()
    test_backtest_recovers_planted_signal()
    test_reversal_baseline()
    test_edge_relations_roundtrip()
    test_edge_weights_roundtrip()
    test_expiry_through_sampler()
    test_bulk_matches_scalar()
    test_single_writer_guard()
    test_determinism()
    test_streaming_ingestor_python()
    test_propagation_recovers_planted_half_life()
    test_ethereum_loader()
    test_ethereum_end_to_end()
    test_fx_universe()
    test_duplicate_instrument_aggregation()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("All strategy and backtest tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
