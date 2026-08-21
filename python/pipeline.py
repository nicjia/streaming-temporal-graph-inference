"""
End-to-end pipeline: GDELT events -> C++ graph -> TGAT -> walk-forward backtest.

    python python/pipeline.py --source synthetic
    python python/pipeline.py --source gdelt --prices data/yfinance/market_data_*.csv

The hypothesis under test is that a country facing rising *expected* conflict
involvement -- as predicted by the temporal graph model, not merely as already
reported -- underperforms its peers over the following days. The harness exists
to measure that claim honestly, including against baselines that use no model,
and a random control that should earn nothing.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

from dataclasses import replace  # noqa: E402

from backtest.engine import (BacktestConfig, METRICS_HEADER, format_metrics,  # noqa: E402
                             run_backtest)
from backtest.evaluate import average_precision, mean_reciprocal_rank, roc_auc  # noqa: E402
from backtest.folds import walk_forward_folds  # noqa: E402
from ingestion.corpus import CONFLICT_CLASSES, load_corpus, synthetic_corpus  # noqa: E402
from ingestion.historical_replayer import DataReplayer  # noqa: E402
from ingestion.id_mapper import EntityMapper  # noqa: E402
from models import (PCSRTemporalSampler, ReturnForecastModel,  # noqa: E402
                    TGATIntensityModel, TGATLinkModel, build_intensity_targets,
                    build_return_targets, forecast_signal,
                    information_coefficient, train_forecast, train_intensity)
from strategy.signals import (event_count_signal, goldstein_signal,  # noqa: E402
                              intensity_signal, model_signal, random_signal,
                              reversal_signal, signal_dates)
from strategy.universe import Universe, etf_universe, fx_universe  # noqa: E402


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_prices(pattern):
    """Load price history written by DataFetcher.fetch_yfinance_data."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(
            f"No price files matched {pattern!r}.\n"
            "Download some, covering the same span as your GDELT slices:\n"
            "  python python/ingestion/data_fetcher.py --start 2024-01-01 "
            "--end 2024-04-01 \\\n"
            "      --tickers SPY FXI EWJ EWG EWU EWQ INDA EWZ EWC EWA EWY EWW EIS TUR\n"
            "Or run the pipeline on synthetic data first:\n"
            "  python python/pipeline.py --source synthetic")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        columns = {c.lower(): c for c in frame.columns}
        if "date" not in columns or "ticker" not in columns or "close" not in columns:
            raise ValueError(f"{path} needs Date, Ticker and Close columns; "
                             f"found {list(frame.columns)}")
        rename = {columns["date"]: "date", columns["ticker"]: "ticker",
                  columns["close"]: "close"}
        wanted = ["date", "ticker", "close"]
        if "volume" in columns:
            rename[columns["volume"]] = "volume"
            wanted.append("volume")
        frames.append(frame.rename(columns=rename)[wanted])

    prices = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date.astype(str)
    return prices.drop_duplicates(subset=["date", "ticker"], keep="last")


def build_graph(events, mapper, label, quiet=False, relations=None):
    """Insert an event table into a fresh PCSR graph and return (graph, sampler)."""
    if not quiet:
        print(f"\nBuilding {label} graph ({len(events):,} events)...")

    replayer = DataReplayer.for_events(num_events=max(len(events), 1024),
                                       num_nodes=mapper.max_entities)
    replayer.replay_events(events["src"].to_numpy(), events["dst"].to_numpy(),
                           events["ts"].to_numpy(), mapper, quiet=quiet,
                           relations=relations)

    sampler = PCSRTemporalSampler(replayer.graph)
    if not sampler.validate():
        raise RuntimeError(f"{label} graph runs are not time-ordered; "
                           "the temporal sampler requires chronological insertion")
    return replayer.graph, sampler


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def subsample_events(events, limit, rng):
    """
    Cap the number of supervised training events, keeping chronological order.

    A year of GDELT is several million conflict events, and every one of them
    costs a two-hop TGAT forward pass. Training on all of them is not more
    correct, just slower -- the encoder still *reads* the full graph through the
    sampler regardless of which events supervise it. Sampling is spread evenly
    across the window rather than taken as a head or tail slice, so the model
    still sees the whole training period.
    """
    if limit <= 0 or len(events) <= limit:
        return events
    positions = np.linspace(0, len(events) - 1, limit).astype(int)
    return events.iloc[np.unique(positions)]


def train_model(model, events, candidate_ids, epochs, batch_size, learning_rate,
                rng, quiet=False):
    """Train the link predictor on one fold's training events."""
    if len(events) < batch_size:
        batch_size = max(8, len(events) // 4)
    if len(events) < 16:
        return []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    src = events["src_id"].to_numpy()
    dst = events["dst_id"].to_numpy()
    ts = events["ts"].to_numpy()

    # Skip the earliest slice of the training window: those events have almost
    # no history behind them, so their attention rows are mostly masked and they
    # contribute little but noise.
    begin = len(src) // 10

    losses = []
    for epoch in range(epochs):
        for offset in range(begin, len(src), batch_size):
            end = min(offset + batch_size, len(src))
            if end - offset < 4:
                continue
            span = end - offset
            negative_dst = rng.choice(candidate_ids, size=span)
            # Source corruption too: without it the model has no reason to make
            # scores comparable between countries, and the cross-sectional
            # signal built from them is meaningless. See TGATLinkModel.loss.
            negative_src = rng.choice(candidate_ids, size=span)

            optimizer.zero_grad()
            loss, _, _ = model.loss(src[offset:end], dst[offset:end],
                                    ts[offset:end], negative_dst, negative_src)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if not quiet and losses:
            window = max(1, len(losses) // epochs)
            print(f"    epoch {epoch + 1}/{epochs}  loss {np.mean(losses[-window:]):.4f}")
    return losses


def degree_prior(events, candidate_ids):
    """
    Empirical in-degree distribution over the training window.

    Used to draw negatives that look like real targets. Without it, negatives
    come from a uniform draw over ~220 countries while real edges concentrate
    massively -- the top five targets take 45% of all conflict edges -- so the
    task degenerates into "is this a country that appears at all", which a
    single popularity number answers. Measured: popularity alone scores 0.915
    AUC against uniform negatives and 0.489 against degree-matched ones.
    """
    counts = np.zeros(int(np.max(candidate_ids)) + 1, dtype=np.float64)
    for target in events["dst_id"].to_numpy():
        counts[target] += 1.0
    prior = counts[candidate_ids]
    total = prior.sum()
    if total <= 0:
        return None
    return prior / total


def cooccurrence_baseline(train_events, num_ids):
    """
    Static count of how often each ordered pair has been seen.

    No model, no time, no attention -- two nested loops. It is here because it
    beat the TGAT on this task, and a baseline that beats your model belongs in
    the report rather than in a footnote.
    """
    table = {}
    for source, target in zip(train_events["src_id"].to_numpy(),
                              train_events["dst_id"].to_numpy()):
        key = (int(source), int(target))
        table[key] = table.get(key, 0) + 1

    def score(sources, targets):
        return np.array([np.log1p(table.get((int(a), int(b)), 0))
                         for a, b in zip(sources, targets)])
    return score


@torch.no_grad()
def evaluate_model(model, events, candidate_ids, rng, negatives_per_positive=10,
                   max_events=2000, prior=None, baseline=None):
    """
    Ranking metrics for one fold's test events, under two negative-sampling
    schemes.

    `auc` is the uniform-negative number the literature usually quotes.
    `auc_degree_matched` is the one that means something here; when the two
    disagree by as much as they do on GDELT, the first is measuring the
    dataset's concentration rather than the model.
    """
    model.eval()
    if events.empty:
        return {}

    if len(events) > max_events:
        events = events.iloc[np.linspace(0, len(events) - 1, max_events).astype(int)]

    src = events["src_id"].to_numpy()
    dst = events["dst_id"].to_numpy()
    ts = events["ts"].to_numpy()

    positive = model.score(src, dst, ts).numpy()

    uniform_columns, matched_columns = [], []
    for _ in range(negatives_per_positive):
        corrupted = rng.choice(candidate_ids, size=len(src))
        uniform_columns.append(model.score(src, corrupted, ts).numpy())
        if prior is not None:
            matched = rng.choice(candidate_ids, size=len(src), p=prior)
            matched_columns.append(model.score(src, matched, ts).numpy())

    uniform = np.stack(uniform_columns, axis=1)
    metrics = {
        "events": int(len(src)),
        "auc": roc_auc(positive, uniform.ravel()),
        "average_precision": average_precision(positive, uniform.ravel()),
        "mrr": mean_reciprocal_rank(positive, uniform),
    }

    if matched_columns:
        matched = np.stack(matched_columns, axis=1)
        metrics["auc_degree_matched"] = roc_auc(positive, matched.ravel())
        metrics["mrr_degree_matched"] = mean_reciprocal_rank(positive, matched)

        if baseline is not None:
            base_pos = baseline(src, dst)
            base_neg = np.concatenate([baseline(src, matched[:, i])
                                       for i in range(matched.shape[1])])
            metrics["baseline_auc_degree_matched"] = roc_auc(base_pos, base_neg)

    return metrics


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args):
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # -- events and prices ------------------------------------------------
    if args.source == "synthetic":
        print("Generating synthetic coupled event/price data "
              "(conflict genuinely drives next-day returns)...")
        events, prices = synthetic_corpus(num_days=args.synthetic_days,
                                          seed=args.seed)
    else:
        try:
            events = load_corpus(args.gdelt_glob, entity_level="country",
                                 cache=args.cache, rebuild=args.rebuild_cache)
        except FileNotFoundError as error:
            raise SystemExit(str(error))
        prices = load_prices(args.prices)

    universe = fx_universe() if args.universe == "fx" else etf_universe()
    if args.min_dollar_volume > 0:
        universe = universe.liquidity_screen(prices, args.min_dollar_volume)
    if args.screen_tradability:
        universe = universe.tradability_screen(prices, max_ann_vol=args.max_ann_vol)
    print(f"\nUniverse: {universe}")

    conflict = events[events["quad_class"].isin(CONFLICT_CLASSES)].reset_index(drop=True)
    print(f"Events: {len(events):,} total, {len(conflict):,} conflict "
          f"({len(conflict) / max(len(events), 1):.1%})")
    if len(conflict) < 200:
        raise SystemExit(
            f"Only {len(conflict)} conflict events -- not enough to train or backtest.\n"
            "Download more history:\n"
            "  python python/ingestion/data_fetcher.py --start 2024-01-01 --end 2024-04-01")

    # -- graph ------------------------------------------------------------
    countries = sorted(set(conflict["src"]) | set(conflict["dst"]))
    mapper = EntityMapper(filepath=args.entity_map, max_entities=len(countries) + 16)

    # GDELT's EventRootCode (1-20: statement, appeal, ..., assault, fight,
    # mass violence) as the edge relation. Without it the graph records that two
    # countries interacted but not how, so the attention cannot weight a threat
    # differently from a negotiation.
    relations = None
    num_relations = 0
    if args.relations != "none":
        column = {"root": "event_root_code", "quad": "quad_class"}[args.relations]
        if column in conflict.columns:
            codes = pd.to_numeric(conflict[column], errors="coerce").fillna(0)
            relations = codes.clip(0, 65534).astype(np.uint16).to_numpy()
            num_relations = int(relations.max()) + 1
            print(f"Edge relations: {args.relations} "
                  f"({num_relations} distinct types, "
                  f"{(relations > 0).mean():.0%} of edges typed)")
        else:
            print(f"Edge relations requested but column {column!r} is not in the "
                  f"corpus; rebuild the cache with --rebuild-cache")

    graph, sampler = build_graph(conflict, mapper, "conflict", relations=relations)

    tradable = [c for c in countries
                if universe.is_tradable(c) and mapper.get_id(c) >= 0]
    tradable_ids = np.array([mapper.get_id(c) for c in tradable], dtype=np.int64)
    candidate_ids = np.array([mapper.get_id(c) for c in countries], dtype=np.int64)
    candidate_ids = candidate_ids[candidate_ids >= 0]
    print(f"Countries in graph: {len(countries)}  |  tradable: {len(tradable)}")

    if len(tradable) < args.min_names:
        raise SystemExit(f"Only {len(tradable)} tradable countries; need at least "
                         f"{args.min_names}. Widen the universe or the data range.")

    conflict = conflict.assign(
        src_id=mapper.get_ids(conflict["src"].to_numpy()),
        dst_id=mapper.get_ids(conflict["dst"].to_numpy()),
    )
    conflict = conflict[(conflict["src_id"] >= 0) & (conflict["dst_id"] >= 0)]

    # -- walk-forward -----------------------------------------------------
    folds = walk_forward_folds(conflict["ts"].to_numpy(), num_folds=args.folds,
                               min_train_fraction=args.min_train_fraction,
                               expanding=not args.rolling)

    model_frames = []
    intensity_frames = []
    forecast_frames = []
    fold_metrics = []

    for fold in folds:
        print(f"\n{fold.describe()}")
        train_events = conflict[(conflict["ts"] >= fold.train_start) &
                                (conflict["ts"] < fold.train_end)]
        test_events = conflict[(conflict["ts"] >= fold.test_start) &
                               (conflict["ts"] < fold.test_end)]
        print(f"  train {len(train_events):,} events | test {len(test_events):,}")
        train_events = subsample_events(train_events, args.max_train_events, rng)
        if args.max_train_events and len(train_events) < len(
                conflict[(conflict["ts"] >= fold.train_start) &
                         (conflict["ts"] < fold.train_end)]):
            print(f"  subsampled to {len(train_events):,} supervised events")

        if len(train_events) < 64 or test_events.empty:
            print("  skipped: not enough events")
            continue

        # A fresh model per fold. Carrying weights forward would mean fold k's
        # model had seen fold k-1's test window during training, which is the
        # leak walk-forward exists to prevent.
        torch.manual_seed(args.seed + fold.index)
        model = TGATLinkModel(mapper.max_entities, sampler,
                              node_dim=args.node_dim, time_dim=args.time_dim,
                              num_layers=args.layers, num_neighbors=args.neighbors,
                              dropout=args.dropout, num_relations=num_relations)

        train_model(model, train_events, candidate_ids, args.epochs,
                    args.batch_size, args.learning_rate, rng)

        prior = degree_prior(train_events, candidate_ids)
        baseline = cooccurrence_baseline(train_events, mapper.max_entities)
        metrics = evaluate_model(model, test_events, candidate_ids, rng,
                                 prior=prior, baseline=baseline)
        metrics["fold"] = fold.index
        fold_metrics.append(metrics)
        print(f"  link prediction: AUC {metrics['auc']:.3f} (uniform negs)  "
              f"AP {metrics['average_precision']:.3f}  MRR {metrics['mrr']:.3f}")
        if "auc_degree_matched" in metrics:
            print(f"    degree-matched negatives:  TGAT {metrics['auc_degree_matched']:.3f}"
                  f"  vs static co-occurrence "
                  f"{metrics.get('baseline_auc_degree_matched', float('nan')):.3f}"
                  f"   <- the comparison that matters")

        dates = signal_dates(fold.test_start, fold.test_end)
        if len(dates) == 0:
            print("  no business days in test window; no signal generated")
            continue

        # Ranking scores are the wrong raw material for a cross-sectional
        # signal (see models/intensity.py), so a second head is trained on the
        # forecast the strategy actually consumes. Both are reported.
        torch.manual_seed(args.seed + 1000 + fold.index)
        intensity_model = TGATIntensityModel(
            mapper.max_entities, sampler,
            node_dim=args.node_dim, time_dim=args.time_dim,
            num_layers=args.layers, num_neighbors=args.neighbors,
            dropout=args.dropout)

        train_dates = signal_dates(fold.train_start, fold.train_end)
        target_nodes, target_times, targets = build_intensity_targets(
            events, tradable, tradable_ids, train_dates,
            horizon_days=args.horizon, cutoff_hour=args.cutoff_hour,
            max_ts=fold.train_end)
        print(f"  intensity training examples: {len(targets):,} "
              f"(mean target {targets.mean():.2f} events/{args.horizon}d)"
              if len(targets) else "  no intensity training examples")

        if len(targets) >= 32:
            train_intensity(intensity_model, target_nodes, target_times, targets,
                            epochs=args.intensity_epochs,
                            batch_size=args.intensity_batch,
                            learning_rate=args.learning_rate, seed=args.seed,
                            quiet=True)
            intensity_frames.append(
                intensity_signal(intensity_model, tradable, tradable_ids, dates,
                                 cutoff_hour=args.cutoff_hour))

        # Phase 1/2: a head that predicts the tradable quantity directly --
        # forward relative return -- rather than an intermediate like conflict
        # intensity. Labels are aligned to the backtest's own timing, and any
        # date whose forward window crosses the training cut-off is dropped.
        torch.manual_seed(args.seed + 2000 + fold.index)
        forecast_model = ReturnForecastModel(
            mapper.max_entities, sampler, task=args.task,
            node_dim=args.node_dim, time_dim=args.time_dim,
            num_layers=args.layers, num_neighbors=args.neighbors,
            dropout=args.dropout)

        r_nodes, r_times, r_targets, _ = build_return_targets(
            prices, universe, tradable, tradable_ids, train_dates,
            horizon_days=args.return_horizon, execution_lag_days=args.lag,
            cutoff_hour=args.cutoff_hour, max_ts=fold.train_end)

        if len(r_targets) >= 32:
            print(f"  return training examples: {len(r_targets):,} "
                  f"(target sd {r_targets.std():.2f})")
            train_forecast(forecast_model, r_nodes, r_times, r_targets,
                           epochs=args.forecast_epochs,
                           batch_size=args.intensity_batch,
                           learning_rate=args.learning_rate,
                           seed=args.seed, quiet=True)
            fold_forecast = forecast_signal(forecast_model, tradable, tradable_ids,
                                            dates, cutoff_hour=args.cutoff_hour)
            forecast_frames.append(fold_forecast)

            # Out-of-sample information coefficient: is the *ordering* right?
            _, _, _, truth = build_return_targets(
                prices, universe, tradable, tradable_ids, dates,
                horizon_days=args.return_horizon, execution_lag_days=args.lag,
                cutoff_hour=args.cutoff_hour)
            if not truth.empty:
                coefficient, days = information_coefficient(fold_forecast, truth)
                fold_metrics[-1]["ic"] = coefficient
                print(f"  return forecast IC: {coefficient:+.4f} over {days} days")
        else:
            print("  not enough return labels to train the forecast head")

        model_frames.append(model_signal(model, sampler, tradable, tradable_ids,
                                         dates, cutoff_hour=args.cutoff_hour))
        print(f"  generated signals for {len(dates)} business days")

    if not model_frames:
        raise SystemExit("No fold produced signals. Try fewer folds or more data.")

    model_panel = pd.concat(model_frames, ignore_index=True)
    intensity_panel = (pd.concat(intensity_frames, ignore_index=True)
                       if intensity_frames else pd.DataFrame())
    forecast_panel = (pd.concat(forecast_frames, ignore_index=True)
                      if forecast_frames else pd.DataFrame())
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(model_panel["date"].unique())))

    # -- backtest ---------------------------------------------------------
    config = BacktestConfig(direction=args.direction,
                            execution_lag_days=args.lag,
                            cost_bps=args.cost_bps,
                            max_weight=args.max_weight,
                            min_names=args.min_names,
                            timeseries_window=args.timeseries_window)

    # Each signal carries its own sign. A conflict forecast is traded short
    # (more expected conflict, worse relative performance); a return forecast is
    # already in return space and is traded long. Applying one global direction
    # to both would guarantee one of them was backwards.
    candidates = {
        "TGAT return forecast": (forecast_panel, +1),
        "TGAT intensity": (intensity_panel, args.direction),
        "TGAT link pressure": (model_panel, args.direction),
        "baseline: goldstein": (goldstein_signal(events, tradable, all_dates,
                                                 args.baseline_window), args.direction),
        "baseline: event count": (event_count_signal(events, tradable, all_dates,
                                                     args.baseline_window), args.direction),
        "baseline: 1d reversal": (reversal_signal(prices, universe, tradable,
                                                  all_dates), +1),
        "control: random": (random_signal(tradable, all_dates, seed=args.seed), +1),
    }

    print("\n" + "=" * len(METRICS_HEADER))
    print(f"BACKTEST  lag={args.lag}d  cost={args.cost_bps}bps  "
          f"direction={args.direction:+d}  names={len(tradable)}")
    print("=" * len(METRICS_HEADER))
    print(METRICS_HEADER)
    print("-" * len(METRICS_HEADER))

    results = {}
    for name, (panel, direction) in candidates.items():
        if panel is None or panel.empty:
            print(f"{name:<22} (no signal produced)")
            continue
        signal_config = replace(config, direction=direction)
        try:
            result = run_backtest(panel, prices, universe, signal_config)
        except ValueError as error:
            print(f"{name:<22} ({error})")
            continue
        results[name] = result
        print(format_metrics(name, result["metrics"]))

    if fold_metrics:
        mean_auc = float(np.mean([m["auc"] for m in fold_metrics]))
        print(f"\nLink prediction across {len(fold_metrics)} folds: "
              f"mean AUC {mean_auc:.3f} (uniform negatives)")
        matched = [m["auc_degree_matched"] for m in fold_metrics
                   if "auc_degree_matched" in m]
        base = [m["baseline_auc_degree_matched"] for m in fold_metrics
                if "baseline_auc_degree_matched" in m]
        if matched:
            print(f"  degree-matched: TGAT {np.mean(matched):.3f}"
                  + (f"  vs static co-occurrence {np.mean(base):.3f}" if base else ""))
        ics = [m["ic"] for m in fold_metrics if "ic" in m and np.isfinite(m["ic"])]
        if ics:
            print(f"Return forecast IC across {len(ics)} folds: mean {np.mean(ics):+.4f} "
                  f"(per fold: {', '.join(f'{v:+.3f}' for v in ics)})")

    print(f"\nCompleted in {time.perf_counter() - started:.1f}s")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for name, result in results.items():
            slug = name.replace(":", "").replace(" ", "_").lower()
            result["daily"].to_csv(os.path.join(args.out, f"daily_{slug}.csv"))
        model_panel.to_csv(os.path.join(args.out, "signal_tgat.csv"), index=False)
        summary = pd.DataFrame({name: result["metrics"] for name, result in results.items()}).T
        summary.to_csv(os.path.join(args.out, "summary.csv"))
        for label, panel in (("intensity", intensity_panel),
                             ("forecast", forecast_panel)):
            if not panel.empty:
                panel.to_csv(os.path.join(args.out, f"signal_{label}.csv"), index=False)
        print(f"Wrote results to {args.out}/")

    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument("--source", choices=["gdelt", "synthetic"], default="synthetic")
    data.add_argument("--gdelt-glob", default="data/gdelt/*.csv")
    data.add_argument("--prices", default="data/yfinance/*.csv")
    data.add_argument("--cache", default="data/corpus_country.parquet")
    data.add_argument("--rebuild-cache", action="store_true")
    data.add_argument("--entity-map", default="data/entity_map_country.json")
    data.add_argument("--synthetic-days", type=int, default=400)

    model = parser.add_argument_group("model")
    model.add_argument("--node-dim", type=int, default=32)
    model.add_argument("--time-dim", type=int, default=32)
    model.add_argument("--layers", type=int, default=2)
    model.add_argument("--neighbors", type=int, default=10)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--relations", choices=["root", "quad", "none"], default="root",
                       help="Edge relation type stored in the graph")
    model.add_argument("--epochs", type=int, default=2)
    model.add_argument("--batch-size", type=int, default=200)
    model.add_argument("--max-train-events", type=int, default=30_000,
                       help="Cap on supervised link-prediction events per fold "
                            "(0 = no cap). The graph is read in full regardless.")
    model.add_argument("--intensity-batch", type=int, default=256)
    model.add_argument("--task", choices=["regression", "classification"],
                       default="regression",
                       help="Return head objective")
    model.add_argument("--forecast-epochs", type=int, default=20)
    model.add_argument("--return-horizon", type=int, default=1,
                       help="Holding period in trading days for the return label")
    model.add_argument("--intensity-epochs", type=int, default=20,
                       help="Epochs for the intensity head. Needs many more "
                            "than the link task: there are only days x countries "
                            "examples, so an epoch is a handful of steps.")
    model.add_argument("--horizon", type=int, default=5,
                       help="Days ahead the intensity head forecasts")
    model.add_argument("--learning-rate", type=float, default=1e-3)

    walk = parser.add_argument_group("walk-forward")
    walk.add_argument("--folds", type=int, default=4)
    walk.add_argument("--min-train-fraction", type=float, default=0.4)
    walk.add_argument("--rolling", action="store_true",
                      help="Fixed-length training window instead of expanding")

    book = parser.add_argument_group("backtest")
    book.add_argument("--direction", type=int, default=-1, choices=[-1, 1],
                      help="-1 shorts countries with high expected conflict")
    book.add_argument("--lag", type=int, default=1, help="Execution lag in trading days")
    book.add_argument("--cost-bps", type=float, default=5.0)
    book.add_argument("--max-weight", type=float, default=0.15)
    book.add_argument("--min-names", type=int, default=5)
    book.add_argument("--universe", choices=["etf", "fx"], default="etf",
                      help="etf = single-country equity ETFs; fx = currencies, "
                           "which trade 24h and are far more liquid")
    book.add_argument("--no-screen-tradability", dest="screen_tradability",
                      action="store_false",
                      help="Keep pegs and crisis currencies in the universe")
    book.add_argument("--max-ann-vol", type=float, default=0.20,
                      help="Drop instruments above this annualised volatility")
    book.add_argument("--min-dollar-volume", type=float, default=2_000_000,
                      help="Drop ETFs below this median daily dollar volume. "
                           "0 disables the screen.")
    book.add_argument("--baseline-window", type=int, default=5)
    book.add_argument("--cutoff-hour", type=int, default=0)
    book.add_argument("--timeseries-window", type=int, default=60,
                      help="Trailing demean window per country; 0 disables")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Directory to write results into")
    return parser


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    main()
