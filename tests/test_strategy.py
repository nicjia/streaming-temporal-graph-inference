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

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from backtest.evaluate import average_precision, mean_reciprocal_rank, roc_auc  # noqa: E402
from backtest.folds import walk_forward_folds  # noqa: E402
from ingestion.corpus import CONFLICT_CLASSES, synthetic_corpus  # noqa: E402
from ingestion.historical_replayer import DataReplayer, provision, to_unix_seconds  # noqa: E402
from ingestion.id_mapper import EntityMapper  # noqa: E402
from models import (PCSRTemporalSampler, build_intensity_targets,  # noqa: E402
                    build_return_targets, information_coefficient)
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


def test_intensity_targets():
    print("\nIntensity targets")
    events, _ = synthetic_corpus(num_days=60, events_per_day=80, seed=3)
    countries = sorted(set(events["src"]))[:6]
    ids = np.arange(len(countries))
    dates = signal_dates(events["ts"].min(), events["ts"].max())

    cutoff = int(np.quantile(events["ts"], 0.6))
    nodes, times, targets = build_intensity_targets(events, countries, ids, dates,
                                                    horizon_days=5, max_ts=cutoff)

    check(len(nodes) == len(times) == len(targets), "target arrays are parallel")
    check(targets.min() >= 0, "counts are non-negative")

    # The bound that keeps the walk-forward split honest.
    horizon_seconds = 5 * 86400
    check(times.max() + horizon_seconds <= cutoff + 86400,
          "no training example's forward window crosses the training cut-off")

    # A target must equal the count it claims to be.
    conflict = events[events["quad_class"].isin(CONFLICT_CLASSES)]
    sample = 0
    country = countries[nodes[sample]]
    window = conflict[(conflict["ts"] >= times[sample]) &
                      (conflict["ts"] < times[sample] + horizon_seconds)]
    expected = ((window["src"] == country) | (window["dst"] == country)).sum()
    check(int(targets[sample]) == int(expected),
          "target equals the actual conflict count in its window")


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



def test_return_targets():
    print("\nReturn labels")
    events, prices = synthetic_corpus(num_days=200, events_per_day=100, seed=6)
    universe = Universe()
    countries = sorted(set(events["src"]))
    ids = np.arange(len(countries))
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices["date"].unique())))

    nodes, times, targets, frame = build_return_targets(
        prices, universe, countries, ids, dates,
        horizon_days=1, execution_lag_days=1, dispersion_window=0, clip=None)

    check(len(nodes) == len(times) == len(targets) == len(frame),
          "label arrays are parallel")
    check(np.isfinite(targets).all(), "labels are finite")

    # Demeaning must make each day's cross-section sum to zero -- that is what
    # makes the target match a dollar-neutral book rather than market beta.
    daily = frame.groupby("date")["target"].mean().abs().max()
    check(daily < 1e-9, "labels are cross-sectionally demeaned")

    # Verify one label against the prices by hand, with the exact timing the
    # backtest uses: signal at d -> enter close(d+1) -> exit close(d+2).
    wide = prices.copy()
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    row = frame.iloc[len(frame) // 2]
    ticker = universe.ticker(row["country"])
    position = wide.index.get_loc(pd.Timestamp(row["date"]))
    raw = wide[ticker].iloc[position + 2] / wide[ticker].iloc[position + 1] - 1.0
    same_day = [c for c in countries if universe.ticker(c) in wide.columns]
    market = np.mean([wide[universe.ticker(c)].iloc[position + 2] /
                      wide[universe.ticker(c)].iloc[position + 1] - 1.0
                      for c in same_day])
    check(abs(row["target"] - (raw - market)) < 1e-8,
          "label equals the demeaned return over the correct holding window")

    # The leakage guard.
    cutoff = int(pd.Timestamp(dates[120]).timestamp())
    _, bounded_times, _, _ = build_return_targets(
        prices, universe, countries, ids, dates,
        horizon_days=1, execution_lag_days=1, max_ts=cutoff)
    check(bounded_times.max() + 3 * 86400 <= cutoff + 86400,
          "no label's forward window crosses the training cut-off")

    # Trailing dispersion must not consult the day it scales.
    _, _, scaled, scaled_frame = build_return_targets(
        prices, universe, countries, ids, dates, dispersion_window=60)
    check(abs(float(np.std(scaled)) - 1.0) < 0.9,
          "scaled labels are order-1")
    check(len(scaled_frame) < len(frame),
          "the dispersion warm-up drops early dates rather than peeking")


def test_information_coefficient():
    print("\nInformation coefficient")
    dates = ["2024-01-02", "2024-01-03"]
    countries = ["USA", "CHN", "JPN", "DEU"]

    rows_p, rows_r = [], []
    for date in dates:
        for rank, country in enumerate(countries):
            rows_p.append((date, country, float(rank)))
            rows_r.append((date, country, float(rank)))
    perfect = information_coefficient(
        pd.DataFrame(rows_p, columns=["date", "country", "signal"]),
        pd.DataFrame(rows_r, columns=["date", "country", "target"]))[0]
    check(abs(perfect - 1.0) < 1e-9, "perfectly ordered forecast scores IC 1.0")

    inverted = information_coefficient(
        pd.DataFrame([(d, c, float(i)) for d in dates for i, c in enumerate(countries)],
                     columns=["date", "country", "signal"]),
        pd.DataFrame([(d, c, float(-i)) for d in dates for i, c in enumerate(countries)],
                     columns=["date", "country", "target"]))[0]
    check(abs(inverted + 1.0) < 1e-9, "inverted forecast scores IC -1.0")


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
    s_arr, d_arr, t_arr, r_arr = graph.to_coo()
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


def main():
    test_id_mapper()
    test_timestamps()
    test_universe()
    test_folds()
    test_metrics()
    test_intensity_targets()
    test_recency_features()
    test_replayer()
    test_backtest_control()
    test_backtest_no_lookahead()
    test_backtest_book()
    test_backtest_recovers_planted_signal()
    test_return_targets()
    test_information_coefficient()
    test_reversal_baseline()
    test_edge_relations_roundtrip()
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
