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

from backtest.engine import (BacktestConfig, METRICS_HEADER, format_metrics,  # noqa: E402
                             run_backtest)
from backtest.evaluate import average_precision, mean_reciprocal_rank, roc_auc  # noqa: E402
from backtest.folds import walk_forward_folds  # noqa: E402
from ingestion.corpus import CONFLICT_CLASSES, load_corpus, synthetic_corpus  # noqa: E402
from ingestion.historical_replayer import DataReplayer  # noqa: E402
from ingestion.id_mapper import EntityMapper  # noqa: E402
from models import PCSRTemporalSampler, TGATLinkModel  # noqa: E402
from strategy.signals import (event_count_signal, goldstein_signal,  # noqa: E402
                              model_signal, random_signal, signal_dates)
from strategy.universe import Universe  # noqa: E402


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_prices(pattern):
    """Load price history written by DataFetcher.fetch_yfinance_data."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No price files matched {pattern!r}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        columns = {c.lower(): c for c in frame.columns}
        if "date" not in columns or "ticker" not in columns or "close" not in columns:
            raise ValueError(f"{path} needs Date, Ticker and Close columns; "
                             f"found {list(frame.columns)}")
        frames.append(frame.rename(columns={columns["date"]: "date",
                                            columns["ticker"]: "ticker",
                                            columns["close"]: "close"})
                      [["date", "ticker", "close"]])

    prices = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    prices["date"] = pd.to_datetime(prices["date"]).dt.date.astype(str)
    return prices.drop_duplicates(subset=["date", "ticker"], keep="last")


def build_graph(events, mapper, label, quiet=False):
    """Insert an event table into a fresh PCSR graph and return (graph, sampler)."""
    if not quiet:
        print(f"\nBuilding {label} graph ({len(events):,} events)...")

    replayer = DataReplayer.for_events(num_events=max(len(events), 1024),
                                       num_nodes=mapper.max_entities)
    replayer.replay_events(events["src"].to_numpy(), events["dst"].to_numpy(),
                           events["ts"].to_numpy(), mapper, quiet=quiet)

    sampler = PCSRTemporalSampler(replayer.graph)
    if not sampler.validate():
        raise RuntimeError(f"{label} graph runs are not time-ordered; "
                           "the temporal sampler requires chronological insertion")
    return replayer.graph, sampler


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

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


@torch.no_grad()
def evaluate_model(model, events, candidate_ids, rng, negatives_per_positive=10,
                   max_events=2000):
    """Ranking metrics for one fold's test events."""
    model.eval()
    if events.empty:
        return {}

    if len(events) > max_events:
        events = events.iloc[np.linspace(0, len(events) - 1, max_events).astype(int)]

    src = events["src_id"].to_numpy()
    dst = events["dst_id"].to_numpy()
    ts = events["ts"].to_numpy()

    positive = model.score(src, dst, ts).numpy()

    negative_columns = []
    for _ in range(negatives_per_positive):
        corrupted = rng.choice(candidate_ids, size=len(src))
        negative_columns.append(model.score(src, corrupted, ts).numpy())
    negatives = np.stack(negative_columns, axis=1)

    return {
        "events": int(len(src)),
        "auc": roc_auc(positive, negatives.ravel()),
        "average_precision": average_precision(positive, negatives.ravel()),
        "mrr": mean_reciprocal_rank(positive, negatives),
    }


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
        events = load_corpus(args.gdelt_glob, entity_level="country",
                             cache=args.cache, rebuild=args.rebuild_cache)
        prices = load_prices(args.prices)

    universe = Universe()
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
    graph, sampler = build_graph(conflict, mapper, "conflict")

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
    fold_metrics = []

    for fold in folds:
        print(f"\n{fold.describe()}")
        train_events = conflict[(conflict["ts"] >= fold.train_start) &
                                (conflict["ts"] < fold.train_end)]
        test_events = conflict[(conflict["ts"] >= fold.test_start) &
                               (conflict["ts"] < fold.test_end)]
        print(f"  train {len(train_events):,} events | test {len(test_events):,}")

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
                              dropout=args.dropout)

        train_model(model, train_events, candidate_ids, args.epochs,
                    args.batch_size, args.learning_rate, rng)

        metrics = evaluate_model(model, test_events, candidate_ids, rng)
        metrics["fold"] = fold.index
        fold_metrics.append(metrics)
        print(f"  link prediction: AUC {metrics['auc']:.3f}  "
              f"AP {metrics['average_precision']:.3f}  MRR {metrics['mrr']:.3f}")

        dates = signal_dates(fold.test_start, fold.test_end)
        if len(dates) == 0:
            print("  no business days in test window; no signal generated")
            continue
        model_frames.append(model_signal(model, sampler, tradable, tradable_ids,
                                         dates, cutoff_hour=args.cutoff_hour))
        print(f"  generated signals for {len(dates)} business days")

    if not model_frames:
        raise SystemExit("No fold produced signals. Try fewer folds or more data.")

    model_panel = pd.concat(model_frames, ignore_index=True)
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(model_panel["date"].unique())))

    # -- backtest ---------------------------------------------------------
    config = BacktestConfig(direction=args.direction,
                            execution_lag_days=args.lag,
                            cost_bps=args.cost_bps,
                            max_weight=args.max_weight,
                            min_names=args.min_names)

    candidates = {
        "TGAT conflict pressure": model_panel,
        "baseline: goldstein": goldstein_signal(events, tradable, all_dates,
                                                args.baseline_window),
        "baseline: event count": event_count_signal(events, tradable, all_dates,
                                                    args.baseline_window),
        "control: random": random_signal(tradable, all_dates, seed=args.seed),
    }

    print("\n" + "=" * len(METRICS_HEADER))
    print(f"BACKTEST  lag={args.lag}d  cost={args.cost_bps}bps  "
          f"direction={args.direction:+d}  names={len(tradable)}")
    print("=" * len(METRICS_HEADER))
    print(METRICS_HEADER)
    print("-" * len(METRICS_HEADER))

    results = {}
    for name, panel in candidates.items():
        if panel.empty:
            print(f"{name:<22} (no signal produced)")
            continue
        try:
            result = run_backtest(panel, prices, universe, config)
        except ValueError as error:
            print(f"{name:<22} ({error})")
            continue
        results[name] = result
        print(format_metrics(name, result["metrics"]))

    if fold_metrics:
        mean_auc = float(np.mean([m["auc"] for m in fold_metrics]))
        print(f"\nLink prediction across {len(fold_metrics)} folds: mean AUC {mean_auc:.3f}")

    print(f"\nCompleted in {time.perf_counter() - started:.1f}s")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for name, result in results.items():
            slug = name.replace(":", "").replace(" ", "_").lower()
            result["daily"].to_csv(os.path.join(args.out, f"daily_{slug}.csv"))
        model_panel.to_csv(os.path.join(args.out, "signal_tgat.csv"), index=False)
        summary = pd.DataFrame({name: result["metrics"] for name, result in results.items()}).T
        summary.to_csv(os.path.join(args.out, "summary.csv"))
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
    model.add_argument("--epochs", type=int, default=2)
    model.add_argument("--batch-size", type=int, default=200)
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
    book.add_argument("--baseline-window", type=int, default=5)
    book.add_argument("--cutoff-hour", type=int, default=0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Directory to write results into")
    return parser


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    main()
