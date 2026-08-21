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

# src_country / dst_country were dropped: at country granularity they are exact
# duplicates of src / dst, and nothing downstream ever read them. Two redundant
# categorical columns is ~10 bytes a row, which is gigabytes across five years.
CORPUS_COLUMNS = ["ts", "src", "dst", "quad_class", "event_root_code", "goldstein",
                  "tone", "num_mentions"]


def _read_cache(path):
    """Read a cached corpus, dispatching on extension."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_cache(table, path):
    """
    Write the corpus cache.

    Parquet is preferred when the extension asks for it, but it needs pyarrow
    or fastparquet, which are not in requirements.txt -- so a missing engine
    falls back to gzipped CSV beside it rather than taking down a pipeline run
    over a cache write.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if path.endswith(".parquet"):
        try:
            table.to_parquet(path, index=False)
            return path
        except ImportError:
            path = path[: -len(".parquet")] + ".csv.gz"

    table.to_csv(path, index=False)
    return path


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


def _reduce(raw, entity_level):
    """Turn a raw GDELT frame into the compact corpus schema."""
    table = pd.DataFrame(index=raw.index)
    table["ts"] = to_unix_seconds(raw["DATEADDED"])

    src_country = _resolve_country(raw.get("Actor1CountryCode"), raw["Actor1Name"])
    dst_country = _resolve_country(raw.get("Actor2CountryCode"), raw["Actor2Name"])

    if entity_level == "country":
        table["src"] = src_country
        table["dst"] = dst_country
    elif entity_level == "actor":
        table["src"] = raw["Actor1Name"].astype(str).str.strip().str.upper()
        table["dst"] = raw["Actor2Name"].astype(str).str.strip().str.upper()
    else:
        raise ValueError(f"entity_level must be 'country' or 'actor', got {entity_level!r}")

    table["quad_class"] = pd.to_numeric(raw.get("QuadClass"), errors="coerce")
    # CAMEO event root code 1-20, ordered roughly by escalation.
    table["event_root_code"] = pd.to_numeric(raw.get("EventRootCode"), errors="coerce")
    table["goldstein"] = pd.to_numeric(raw.get("GoldsteinScale"), errors="coerce")
    table["tone"] = pd.to_numeric(raw.get("AvgTone"), errors="coerce")
    table["num_mentions"] = pd.to_numeric(raw.get("NumMentions"), errors="coerce").fillna(1)

    before = len(table)
    table = table.dropna(subset=["ts", "src", "dst"])
    table = table[table["src"] != table["dst"]]  # self-loops carry no relation

    table = table[CORPUS_COLUMNS]

    # Narrow dtypes. Across five years (~44M rows) the difference between the
    # obvious types and these is roughly 2 GB of resident memory, which decides
    # whether the corpus loads at all on a 16 GB machine.
    #   ts               int32   -- unix seconds, good to 2038
    #   src / dst        category -- ~220 distinct countries, one byte a code
    #   quad_class       uint8   -- 1..4, 0 means unknown
    #   event_root_code  uint8   -- 1..20, 0 means unknown (== RELATION_UNKNOWN)
    table["ts"] = table["ts"].astype("int32")
    for column in ("src", "dst"):
        table[column] = table[column].astype("category")
    for column in ("quad_class", "event_root_code"):
        table[column] = table[column].fillna(0).clip(0, 255).astype("uint8")
    for column in ("goldstein", "tone", "num_mentions"):
        table[column] = table[column].astype("float32")

    return table, before - len(table)


def load_corpus(pattern="data/gdelt/*.csv", entity_level="country",
                cache=None, rebuild=False, quiet=False, batch_size=500,
                progress_every=4):
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
    if cache and not rebuild:
        for candidate in (cache, cache.replace(".parquet", ".csv.gz")):
            if os.path.exists(candidate):
                table = _read_cache(candidate)
                if not quiet:
                    print(f"Loaded {len(table):,} events from cache {candidate}")
                return table

    paths = sorted(glob.glob(pattern))
    if pattern.endswith(".csv"):
        # Slices are stored gzipped; accept both so a plain-.csv glob still
        # finds everything.
        paths = sorted(set(paths) | set(glob.glob(pattern + ".gz")))
    if not paths:
        raise FileNotFoundError(
            f"No GDELT slices matched {pattern!r}. Download some first:\n"
            f"  python python/ingestion/data_fetcher.py --start 2024-01-01 --end 2024-02-01")

    if not quiet:
        print(f"Reading {len(paths)} GDELT slice(s)...")

    # Processed in batches rather than concatenated raw. A year of 15-minute
    # slices is ~44M rows across 16 mostly-string columns; holding all of that
    # in pandas before reducing it needs tens of gigabytes, while the reduced
    # form is ~10M rows of nine narrow columns. Reduce first, accumulate second.
    chunks = []
    dropped = 0
    read = 0

    for batch_start in range(0, len(paths), batch_size):
        frames = []
        for path in paths[batch_start:batch_start + batch_size]:
            frame = pd.read_csv(path, low_memory=False)
            if "Actor1Name" in frame.columns:
                frames.append(frame)
        if not frames:
            continue

        raw = pd.concat(frames, ignore_index=True)
        del frames
        raw = raw.dropna(subset=["Actor1Name", "Actor2Name", "DATEADDED"])
        read += len(raw)

        reduced, lost = _reduce(raw, entity_level)
        dropped += lost
        chunks.append(reduced)
        del raw

        if not quiet and progress_every and \
                (batch_start // batch_size) % progress_every == 0:
            done = min(batch_start + batch_size, len(paths))
            kept = sum(len(c) for c in chunks)
            print(f"  {done}/{len(paths)} slices, {kept:,} events kept", flush=True)

    if not chunks:
        raise ValueError("no usable rows found in the matched slices")

    table = pd.concat(chunks, ignore_index=True)
    del chunks

    table = table.sort_values("ts", kind="stable").reset_index(drop=True)
    table["ts"] = table["ts"].astype("int64")  # widen back for arithmetic

    if not quiet:
        span = pd.to_datetime(table["ts"], unit="s")
        print(f"Corpus: {len(table):,} events ({dropped:,} dropped as "
              f"unresolvable or self-referential)")
        if len(table):
            print(f"  span {span.min()} -> {span.max()}, "
                  f"{table['src'].nunique():,} distinct sources")

    if cache:
        written = _write_cache(table, cache)
        if not quiet:
            print(f"  cached to {written}")

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
                int(rng.integers(14, 21) if is_conflict else rng.integers(1, 6)),
                float(rng.normal(-5 if is_conflict else 4, 2)),
                float(rng.normal(-3 if is_conflict else 2, 3)),
                float(rng.integers(1, 20)),
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
