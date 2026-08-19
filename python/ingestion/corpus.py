"""
Consolidates downloaded GDELT slices into one chronologically ordered event
table, resolved to whichever entity granularity the caller wants.
"""

import glob
import os

import numpy as np
import pandas as pd

from ingestion.historical_replayer import to_unix_seconds
from strategy.universe import Universe

# GDELT's QuadClass. Splitting conflict from cooperation is what lets the model
# predict something directional: the PCSR edge record is (target, timestamp)
# with no relation type, so the only way to distinguish "will fight" from "will
# cooperate" is to build a separate graph per class.
QUAD_VERBAL_COOP = 1
QUAD_MATERIAL_COOP = 2
QUAD_VERBAL_CONFLICT = 3
QUAD_MATERIAL_CONFLICT = 4

CONFLICT_CLASSES = (QUAD_VERBAL_CONFLICT, QUAD_MATERIAL_CONFLICT)
COOPERATION_CLASSES = (QUAD_VERBAL_COOP, QUAD_MATERIAL_COOP)

CORPUS_COLUMNS = ["ts", "src", "dst", "quad_class", "goldstein", "tone",
                  "num_mentions", "src_country", "dst_country"]


def _resolve_country(codes, names):
    """
    ISO3 per row, preferring GDELT's own country code and falling back to the
    actor name.

    The fallback is not cosmetic: only ~59% of rows carry a country code, and
    slices downloaded before this project widened its column set have no code
    column at all.
    """
    resolved = pd.Series(pd.NA, index=names.index, dtype="object")

    if codes is not None:
        cleaned = codes.astype("string").str.strip().str.upper()
        resolved = cleaned.where(cleaned.str.len() == 3)

    missing = resolved.isna()
    if missing.any():
        from_names = names[missing].map(Universe.country_from_name)
        resolved.loc[missing] = from_names

    return resolved


def load_corpus(pattern="data/gdelt/*.csv", entity_level="country",
                cache=None, rebuild=False, quiet=False):
    """
    Build (or load) the consolidated event table.

    Args:
        pattern: Glob over per-slice CSVs written by DataFetcher.
        entity_level: "country" resolves both actors to ISO3 and drops rows
            where either side is unresolvable -- a smaller, denser graph that
            maps directly onto the tradable universe. "actor" keeps the raw
            actor names, giving the large sparse graph that stresses the
            engine but does not map to instruments.
        cache: Optional parquet path. Written on build, read on subsequent
            calls unless rebuild is set.
        rebuild: Ignore any cache and re-read the slices.

    Returns:
        DataFrame with CORPUS_COLUMNS, sorted by ts.
    """
    if cache and os.path.exists(cache) and not rebuild:
        table = pd.read_parquet(cache)
        if not quiet:
            print(f"Loaded {len(table):,} events from cache {cache}")
        return table

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No GDELT slices matched {pattern!r}. Download some first:\n"
            f"  python python/ingestion/data_fetcher.py --start 2024-01-01 --end 2024-02-01")

    if not quiet:
        print(f"Reading {len(paths)} GDELT slice(s)...")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        if "Actor1Name" not in frame.columns:
            continue
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["Actor1Name", "Actor2Name", "DATEADDED"])

    table = pd.DataFrame(index=raw.index)
    table["ts"] = to_unix_seconds(raw["DATEADDED"])

    src_country = _resolve_country(raw.get("Actor1CountryCode"), raw["Actor1Name"])
    dst_country = _resolve_country(raw.get("Actor2CountryCode"), raw["Actor2Name"])
    table["src_country"] = src_country
    table["dst_country"] = dst_country

    if entity_level == "country":
        table["src"] = src_country
        table["dst"] = dst_country
    elif entity_level == "actor":
        table["src"] = raw["Actor1Name"].astype(str).str.strip().str.upper()
        table["dst"] = raw["Actor2Name"].astype(str).str.strip().str.upper()
    else:
        raise ValueError(f"entity_level must be 'country' or 'actor', got {entity_level!r}")

    table["quad_class"] = pd.to_numeric(raw.get("QuadClass"), errors="coerce")
    table["goldstein"] = pd.to_numeric(raw.get("GoldsteinScale"), errors="coerce")
    table["tone"] = pd.to_numeric(raw.get("AvgTone"), errors="coerce")
    table["num_mentions"] = pd.to_numeric(raw.get("NumMentions"), errors="coerce").fillna(1)

    before = len(table)
    table = table.dropna(subset=["ts", "src", "dst"])
    table = table[table["src"] != table["dst"]]  # self-loops carry no relation

    table = table[CORPUS_COLUMNS].sort_values("ts", kind="stable").reset_index(drop=True)
    table["ts"] = table["ts"].astype("int64")

    if not quiet:
        span = pd.to_datetime(table["ts"], unit="s")
        print(f"Corpus: {len(table):,} events ({before - len(table):,} dropped as "
              f"unresolvable or self-referential)")
        if len(table):
            print(f"  span {span.min()} -> {span.max()}, "
                  f"{table['src'].nunique():,} distinct sources")

    if cache:
        directory = os.path.dirname(cache)
        if directory:
            os.makedirs(directory, exist_ok=True)
        table.to_parquet(cache, index=False)
        if not quiet:
            print(f"  cached to {cache}")

    return table


def split_by_quad(table):
    """Separate conflict from cooperation events."""
    conflict = table[table["quad_class"].isin(CONFLICT_CLASSES)].reset_index(drop=True)
    cooperation = table[table["quad_class"].isin(COOPERATION_CLASSES)].reset_index(drop=True)
    return conflict, cooperation


def synthetic_corpus(num_days=180, countries=None, seed=0, events_per_day=400,
                     shock_strength=3.0, num_blocs=3, bloc_purity=0.85,
                     num_countries=24):
    """
    A GDELT-shaped stream with signal deliberately planted in it.

    Needed for two reasons. The obvious one is that the harness has to be
    runnable and testable without months of downloads. The important one is
    that a backtester which cannot recover a signal you know is there is a
    broken backtester, and you cannot establish that on real data, where you do
    not know the answer.

    Two distinct structures are planted, because the model and the baselines
    are sensitive to different things:

    * **Intensity.** Each country carries a slow-moving latent conflict level
      that drives both how often it appears as an aggressor and how likely its
      events are to be conflictual. This is what the trailing-window baselines
      pick up, and it is what drives returns.

    * **Affinity.** Countries belong to blocs. Conflict runs mostly *between*
      blocs and cooperation mostly *within* them, so which partner an event
      lands on is predictable from graph structure. This is what the link
      predictor can learn and the baselines cannot -- without it the generator
      would test the baselines and nothing else.

    Next-day returns are driven by the latent intensity, with noise sized so it
    dominates the variance. The planted edge is real but not free.

    Returns:
        (events DataFrame in corpus format, prices DataFrame [date, ticker, close])
    """
    rng = np.random.default_rng(seed)
    universe = Universe()
    countries = countries or universe.countries()[:num_countries]
    tickers = [universe.ticker(c) for c in countries]
    count = len(countries)

    bloc = np.arange(count) % num_blocs
    same_bloc = bloc[:, None] == bloc[None, :]
    np.fill_diagonal(same_bloc, False)

    # Partner distributions, one row per source: conflict looks outward,
    # cooperation looks inward.
    def partner_weights(prefer_same):
        preferred = same_bloc if prefer_same else ~same_bloc
        np.fill_diagonal(preferred, False)
        weights = np.where(preferred, bloc_purity, 1.0 - bloc_purity).astype(float)
        np.fill_diagonal(weights, 0.0)
        return weights / weights.sum(axis=1, keepdims=True)

    conflict_partners = partner_weights(prefer_same=False)
    cooperation_partners = partner_weights(prefer_same=True)

    start = pd.Timestamp("2022-01-03")
    days = pd.bdate_range(start, periods=num_days)

    latent = rng.normal(0, 1, size=(num_days, count))
    for day in range(1, num_days):
        latent[day] = 0.85 * latent[day - 1] + 0.53 * latent[day]

    rows = []
    for day_index, day in enumerate(days):
        base = pd.Timestamp(day).timestamp()
        source_weights = np.exp(latent[day_index])
        source_weights /= source_weights.sum()
        conflict_probability = 1.0 / (1.0 + np.exp(-latent[day_index]))

        sources = rng.choice(count, size=events_per_day, p=source_weights)
        conflictual = rng.random(events_per_day) < conflict_probability[sources]

        for source, is_conflict in zip(sources, conflictual):
            table = conflict_partners if is_conflict else cooperation_partners
            target = rng.choice(count, p=table[source])
            quad = rng.choice(CONFLICT_CLASSES if is_conflict else COOPERATION_CLASSES)
            rows.append((
                int(base + rng.integers(0, 86400)),
                countries[source], countries[target], int(quad),
                float(rng.normal(-5 if is_conflict else 4, 2)),
                float(rng.normal(-3 if is_conflict else 2, 3)),
                float(rng.integers(1, 20)),
                countries[source], countries[target],
            ))

    events = pd.DataFrame(rows, columns=CORPUS_COLUMNS)
    events = events.sort_values("ts", kind="stable").reset_index(drop=True)

    price_rows = []
    log_price = np.zeros(count)
    for day_index, day in enumerate(days):
        shock = -shock_strength * 0.001 * latent[day_index]
        noise = rng.normal(0, 0.011, size=count)
        log_price = log_price + shock + noise
        for country_index, ticker in enumerate(tickers):
            price_rows.append((day.date().isoformat(), ticker,
                               float(100 * np.exp(log_price[country_index]))))

    prices = pd.DataFrame(price_rows, columns=["date", "ticker", "close"])
    return events, prices
