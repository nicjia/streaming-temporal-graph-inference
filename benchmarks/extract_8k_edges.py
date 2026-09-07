"""
Turn 8-K filings into typed graph edges, in two stages:

  stage 1  fetch    SEC-rate-limited at ~8 req/s, not compute-bound. Produces a
                    gzipped JSONL of cleaned filing text (~50 MB/quarter).
  stage 2  extract  inference-bound and embarrassingly parallel. Reads that
                    file, needs no network or pandas, and shards across cores or
                    array tasks -- so it runs on cluster nodes with no outbound
                    network.

    # stage 1, on a machine with internet
    python benchmarks/extract_8k_edges.py fetch --year 2024 --quarter 2 --dense-only

    # stage 2, against a local or remote llama-server
    python benchmarks/extract_8k_edges.py extract \\
        --input data/edgar/staged/2024Q2.jsonl.gz \\
        --backend http --model qwen --base-url http://127.0.0.1:8080/v1

    # stage 2, as an SGE array job -- same script, same flags
    #$ -t 1-50
    python3 benchmarks/extract_8k_edges.py extract \\
        --input 2024Q2.jsonl.gz --shard $((SGE_TASK_ID-1)) --num-shards 50 \\
        --backend http --model qwen --base-url http://127.0.0.1:$PORT/v1

Output is JSONL, one record per filing, and every run is resumable: a shard
re-reads its own output and skips filings already done. A 12-hour wall clock
limit against a run that might exceed it makes that not optional.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from ingestion.llm_extract import build_backend  # noqa: E402

# pandas and ingestion.edgar are imported inside stage_fetch, not here.
# The extract stage runs on cluster nodes whose module Python has numpy and
# requests and nothing else, and a module-level import of pandas would make the
# whole script unrunnable there for the sake of a stage that never executes.

# Items carrying economic content worth spending inference on. 7.01 (Reg FD)
# and 8.01 (other events) are the open-ended ones and are where most non-earnings
# news actually lands, so they stay in despite being noisy.
DENSE_ITEMS = frozenset({"1.01", "1.02", "2.01", "2.02", "2.05", "2.06",
                         "3.01", "4.02", "5.01", "5.02", "7.01", "8.01"})


def stage_fetch(args) -> None:
    import pandas as pd

    from ingestion.edgar import load_filings

    frames = []
    for year, quarter in _quarters(args.year, args.quarter, args.through_year,
                                   args.through_quarter):
        print(f"[fetch] {year}Q{quarter}", flush=True)
        frame = load_filings(year, quarter, limit=args.limit,
                             data_dir=args.data_dir)
        if frame.empty:
            print(f"  no filings for {year}Q{quarter}")
            continue
        frame["year"] = year
        frame["quarter"] = quarter
        frames.append(frame)
        print(f"  {len(frame):,} filings, "
              f"{frame['text'].str.len().mean():,.0f} chars mean", flush=True)

    if not frames:
        print("nothing fetched")
        return

    everything = pd.concat(frames, ignore_index=True)
    if args.dense_only:
        keep = everything["items"].apply(
            lambda items: (not items) or bool(DENSE_ITEMS.intersection(items)))
        print(f"dense-item filter keeps {int(keep.sum()):,} of {len(everything):,}")
        everything = everything[keep].reset_index(drop=True)

    # Gzipped JSONL, not parquet. The cluster this is destined for has numpy and
    # requests in its module Python and nothing else -- no pandas, no pyarrow --
    # and a staging format that needs a dependency the consumer cannot install
    # is not a staging format. JSONL also streams, so a 12-hour array task never
    # holds the whole corpus in memory, and it compresses to roughly the same
    # size as parquet on text this repetitive.
    out = Path(args.out or f"{args.data_dir}/staged/"
                           f"{args.year}Q{args.quarter}.jsonl.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        for row in everything.itertuples(index=False):
            handle.write(json.dumps({
                "cik": int(row.cik),
                "ticker": row.ticker,
                "company": row.company,
                "date_filed": f"{row.date_filed:%Y-%m-%d}",
                "items": list(row.items),
                "text": row.text,
            }) + "\n")

    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {len(everything):,} filings to {out} ({size_mb:.1f} MB gzipped)")


def stage_annotate(args) -> None:
    """
    Tag each staged filing with the other listed companies its text names.

    Run locally, where pandas can build the CIK->ticker index, so the cluster
    never needs it. The annotation is what turns a pile of stars into a graph:
    on the first full quarter the extractor produced 20,439 edges of which only
    325 listed companies ended up linked to another listed company, because the
    model preferentially names whatever entity is nearest in the sentence --
    usually a private subsidiary, a product or an executive. Handing it the
    listed names a dictionary scan already found redirects that attention at
    almost no cost.

    Hub names are dropped first. A company appearing in a large share of all
    filings is infrastructure -- an exchange, a transfer agent -- not a
    counterparty, and including it would suggest a relationship to Nasdaq in
    every prompt.
    """
    from ingestion.edgar import company_tickers
    from ingestion.entity_resolve import build_name_index, find_listed

    index = build_name_index(company_tickers())
    print(f"name index: {len(index):,} listed companies")

    rows = []
    opener = gzip.open if str(args.input).endswith(".gz") else open
    with opener(args.input, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"filings:    {len(rows):,}")

    # First pass to find the hubs, second to annotate. Document frequency can
    # only be known after seeing every filing.
    raw_hits = []
    frequency: dict[str, int] = {}
    for row in rows:
        hits = find_listed(row["text"], index, exclude=row.get("ticker"))
        raw_hits.append(hits)
        for ticker in hits:
            frequency[ticker] = frequency.get(ticker, 0) + 1

    cutoff = max(2, int(args.max_document_frequency * len(rows)))
    hubs = {t for t, n in frequency.items() if n >= cutoff}
    print(f"dropping {len(hubs)} hub names seen in >={cutoff} filings: "
          f"{', '.join(sorted(hubs)[:10])}")

    name_by_ticker = {}
    for key, ticker in index.items():
        name_by_ticker.setdefault(ticker, key)

    annotated = 0
    out = Path(args.out or str(args.input).replace(".jsonl.gz", ".cand.jsonl.gz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        for row, hits in zip(rows, raw_hits):
            kept = [t for t in hits if t not in hubs]
            if kept:
                annotated += 1
            # Feed the model readable names, not tickers: the text says
            # "Siemens AG", and asking it to match a symbol it never saw is a
            # second inference problem layered on the first.
            row["candidates"] = [name_by_ticker.get(t, t).title() for t in kept]
            row["candidate_tickers"] = kept
            handle.write(json.dumps(row) + "\n")

    print(f"\n{annotated:,} of {len(rows):,} filings name another listed company "
          f"({100*annotated/max(len(rows),1):.1f}%)")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


def _quarters(year, quarter, through_year, through_quarter):
    """Inclusive (year, quarter) range."""
    through_year = through_year or year
    through_quarter = through_quarter or quarter
    out = []
    y, q = year, quarter
    while (y, q) <= (through_year, through_quarter):
        out.append((y, q))
        q += 1
        if q > 4:
            y, q = y + 1, 1
    return out


def _filing_key(cik: int, date: str, text: str) -> str:
    """
    Stable identity for one filing, used to resume a killed run.

    The digest is hashlib, not the builtin hash(): Python randomises string
    hashing per process unless PYTHONHASHSEED is pinned, so a builtin-hash key
    changes on every invocation. That made resume a no-op and every re-run
    append a second copy of work it had already done -- silently, because both
    copies looked like distinct filings.
    """
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]
    day = str(date)[:10].replace("-", "")
    return f"{cik}:{day}:{digest}"


def _done_keys(path: Path) -> set[str]:
    """Filing keys already written, so a killed run resumes where it stopped."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue  # a torn final line from a killed run
    return done


def _read_staged(path: str, shard: int, num_shards: int) -> list[dict]:
    """
    Read the staged corpus, keeping only this shard's rows.

    Stdlib only -- no pandas. This function runs on cluster nodes where the
    module Python has numpy and requests and nothing else.

    Sharding is by line position with a stride, so every task derives the same
    partition from the same file without coordinating, and shard membership
    does not depend on how far any task got.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if num_shards > 1 and index % num_shards != shard:
                continue
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stage_extract(args) -> None:
    rows = _read_staged(args.input, args.shard, args.num_shards)

    base = str(args.input)
    for suffix in (".jsonl.gz", ".jsonl", ".parquet"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    out = Path(args.out or f"{base}.edges.{args.shard:04d}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)

    if args.limit:
        rows = rows[: args.limit]

    print(f"shard {args.shard}/{args.num_shards}: {len(rows):,} filings, "
          f"{len(done):,} already done -> {out}", flush=True)

    extra = {}
    if args.base_url and args.backend != "llama-cpp":
        extra["base_url"] = args.base_url
        extra["timeout"] = args.timeout
    backend = build_backend(args.backend, args.model, max_chars=args.max_chars, **extra)

    pending = [r for r in rows
               if _filing_key(int(r["cik"]), r["date_filed"], r["text"]) not in done]
    started = time.time()
    counters = {"processed": 0, "failed": 0, "edges": 0}
    lock = threading.Lock()

    def handle_one(row):
        """One filing. Returns the record to write; never raises."""
        try:
            edges = backend.extract(row["text"], filer=row.get("company"),
                                    items=row.get("items"),
                                    candidates=row.get("candidates"))
            error = None
        except Exception as exc:
            # One filing that trips the model must not end a 12-hour job.
            edges, error = [], exc
        return row, edges, error

    with out.open("a") as handle:
        def write(row, edges, error):
            with lock:
                if error is not None:
                    counters["failed"] += 1
                    print(f"  [warn] {row.get('ticker')} {row['date_filed']}: "
                          f"{type(error).__name__}: {str(error)[:140]}", flush=True)
                handle.write(json.dumps({
                    "key": _filing_key(int(row["cik"]), row["date_filed"], row["text"]),
                    "cik": int(row["cik"]),
                    "ticker": row.get("ticker"),
                    "company": row.get("company"),
                    "date_filed": row["date_filed"],
                    "items": row.get("items", []),
                    "edges": edges,
                }) + "\n")
                handle.flush()  # cheap next to inference, survives a kill
                counters["processed"] += 1
                counters["edges"] += len(edges)
                if counters["processed"] % args.report_every == 0:
                    rate = counters["processed"] / (time.time() - started)
                    left = (len(pending) - counters["processed"]) / max(rate, 1e-9)
                    print(f"  {counters['processed']:,}/{len(pending):,}  "
                          f"{rate:.3f} filings/s  {counters['edges']:,} edges  "
                          f"{counters['failed']} failed  eta {left/3600:.1f}h",
                          flush=True)

        if args.concurrency > 1:
            # The server is configured with several slots and continuous
            # batching; a strictly sequential client leaves all of them idle but
            # one. Requests are pure I/O waits from Python's side, so threads
            # are the right tool here despite the GIL -- the work happens in the
            # server process.
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for row, edges, error in pool.map(handle_one, pending):
                    write(row, edges, error)
        else:
            for row in pending:
                write(*handle_one(row))

    backend.close()
    elapsed = time.time() - started
    print(f"\ndone: {counters['processed']:,} filings in {elapsed/60:.1f} min "
          f"({counters['processed']/max(elapsed,1e-9):.3f}/s), "
          f"{counters['edges']:,} edges, {counters['failed']} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    fetch = sub.add_parser("fetch", help="download and stage filings (needs network)")
    fetch.add_argument("--year", type=int, required=True)
    fetch.add_argument("--quarter", type=int, required=True, choices=[1, 2, 3, 4])
    fetch.add_argument("--through-year", type=int)
    fetch.add_argument("--through-quarter", type=int, choices=[1, 2, 3, 4])
    fetch.add_argument("--limit", type=int, help="filings per quarter, for testing")
    fetch.add_argument("--dense-only", action="store_true",
                       help="keep only economically dense item codes")
    fetch.add_argument("--data-dir", default="data/edgar")
    fetch.add_argument("--out")
    fetch.set_defaults(func=stage_fetch)

    ann = sub.add_parser("annotate", help="tag staged filings with listed-company mentions")
    ann.add_argument("--input", required=True)
    ann.add_argument("--out")
    ann.add_argument("--max-document-frequency", type=float, default=0.01)
    ann.set_defaults(func=stage_annotate)

    extract = sub.add_parser("extract", help="run the model over a staged corpus")
    extract.add_argument("--input", required=True)
    extract.add_argument("--backend", default="http",
                         choices=["llama-cpp", "http", "server", "openai", "null"])
    extract.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                         help="OpenAI-compatible endpoint for the http backend")
    extract.add_argument("--model", help="GGUF path, or hosted model name")
    extract.add_argument("--shard", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--max-chars", type=int, default=6000)
    extract.add_argument("--limit", type=int)
    extract.add_argument("--report-every", type=int, default=25)
    extract.add_argument("--timeout", type=int, default=900,
                         help="per-request read timeout in seconds")
    extract.add_argument("--concurrency", type=int, default=1,
                         help="in-flight requests; match the server's --parallel")
    extract.add_argument("--out")
    extract.set_defaults(func=stage_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
