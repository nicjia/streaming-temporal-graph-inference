"""
Data acquisition: GDELT 2.0 event exports and Yahoo Finance price history.
"""

import io
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"

# GDELT 2.0 export files are headerless TSVs with 61 columns. These are the
# ones this project uses; the indices are positional and fixed by GDELT's
# documented schema.
GDELT_COLUMNS = {
    0: "GLOBALEVENTID",
    1: "SQLDATE",
    5: "Actor1Code",
    6: "Actor1Name",
    7: "Actor1CountryCode",
    15: "Actor2Code",
    16: "Actor2Name",
    17: "Actor2CountryCode",
    26: "EventCode",
    28: "EventRootCode",
    29: "QuadClass",       # 1 verbal coop, 2 material coop, 3 verbal conflict, 4 material conflict
    30: "GoldsteinScale",  # -10 (most conflictual) .. +10 (most cooperative)
    31: "NumMentions",
    34: "AvgTone",
    59: "DATEADDED",
    60: "SOURCEURL",
}

USER_AGENT = "high-throughput-graph-engine/0.1 (research; contact via repo)"


def gdelt_stamps(start, end):
    """
    Every 15-minute GDELT stamp in [start, end), as YYYYMMDDHHMMSS strings.

    GDELT publishes at :00, :15, :30 and :45 past each hour, so a full day is
    96 files. Not every stamp exists -- the feed has gaps -- which is why the
    downloader treats a 404 as "skip", not "fail".
    """
    start = pd.Timestamp(start).to_pydatetime().replace(tzinfo=None)
    end = pd.Timestamp(end).to_pydatetime().replace(tzinfo=None)

    minute = (start.minute // 15) * 15
    cursor = start.replace(minute=minute, second=0, microsecond=0)

    stamps = []
    while cursor < end:
        stamps.append(cursor.strftime("%Y%m%d%H%M%S"))
        cursor += timedelta(minutes=15)
    return stamps


class DataFetcher:
    def __init__(self, output_dir="data"):
        """Initialises the fetcher and ensures the local data directories exist."""
        self.output_dir = output_dir
        self.gdelt_dir = os.path.join(output_dir, "gdelt")
        self.yf_dir = os.path.join(output_dir, "yfinance")

        os.makedirs(self.gdelt_dir, exist_ok=True)
        os.makedirs(self.yf_dir, exist_ok=True)

        # One Session per thread. Sessions are not documented as thread-safe,
        # and sharing one across the download pool risks corrupted connection
        # reuse under load.
        self._local = threading.local()

    @property
    def session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            self._local.session = session
        return session

    # -- Yahoo Finance ----------------------------------------------------

    def fetch_yfinance_data(self, tickers, start_date, end_date, interval="1d",
                            label=None):
        """
        Downloads historical market data for the specified tickers.

        `label` distinguishes files covering the same dates. Without it two
        different baskets over one date range collide on the filename and the
        second silently overwrites the first.
        """
        import yfinance as yf

        print(f"Fetching Yahoo Finance data for {len(tickers)} tickers "
              f"from {start_date} to {end_date}...")

        df = yf.download(tickers, start=start_date, end=end_date, interval=interval,
                         auto_adjust=True, progress=False)

        if len(tickers) > 1:
            df = df.stack(level=1, future_stack=True).rename_axis(["Date", "Ticker"]).reset_index()
        else:
            df = df.reset_index()
            df["Ticker"] = tickers[0]

        suffix = f"_{label}" if label else ""
        output_path = os.path.join(
            self.yf_dir, f"market_data{suffix}_{start_date}_{end_date}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved market data to {output_path}")
        return df

    # -- GDELT ------------------------------------------------------------

    def slice_path(self, datetime_str):
        """
        Where a slice lives on disk.

        Gzipped: the parsed 16-column slice is ~300 KB plain, and a year of
        15-minute files is ~11 GB at that rate. Compression takes it to roughly
        a fifth of that, and pandas reads .csv.gz transparently, so nothing
        downstream needs to care.
        """
        return os.path.join(self.gdelt_dir, f"gdelt_{datetime_str}.csv.gz")

    def existing_slice(self, datetime_str):
        """Path to an already-downloaded slice in either format, or None."""
        for candidate in (self.slice_path(datetime_str),
                          os.path.join(self.gdelt_dir, f"gdelt_{datetime_str}.csv")):
            if os.path.exists(candidate):
                return candidate
        return None

    def fetch_gdelt_hourly_export(self, datetime_str, overwrite=False, timeout=60):
        """
        Downloads and unzips one GDELT 2.0 export file.

        Format expected: YYYYMMDDHHMMSS (e.g. '20231101080000').
        Returns the parsed DataFrame, or None if the file does not exist.
        """
        cached = self.existing_slice(datetime_str)
        if cached and not overwrite:
            return pd.read_csv(cached, low_memory=False)
        output_path = self.slice_path(datetime_str)

        url = f"{GDELT_BASE}/{datetime_str}.export.CSV.zip"
        response = self.session.get(url, timeout=timeout)

        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise requests.HTTPError(f"{url} returned HTTP {response.status_code}")

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            name = archive.namelist()[0]
            with archive.open(name) as handle:
                indices = sorted(GDELT_COLUMNS)
                df = pd.read_csv(handle, sep="\t", header=None,
                                 usecols=indices,
                                 names=[GDELT_COLUMNS[i] for i in indices],
                                 low_memory=False)

        df.to_csv(output_path, index=False)
        return df

    def fetch_gdelt_range(self, start, end, overwrite=False, pause=0.05,
                          max_retries=3, progress_every=200, max_workers=6):
        """
        Download every 15-minute slice in [start, end).

        Resumable: files already on disk are skipped, so re-running after an
        interruption picks up where it left off.

        Downloads run on a small thread pool because the work is entirely
        I/O-bound -- a single connection spends nearly all its time waiting, and
        sequentially a year takes about three hours against roughly thirty
        minutes at six workers. Six is deliberately modest: these are static
        files on someone else's free public server, and each worker still pauses
        between its own requests. Set max_workers=1 for strictly sequential
        behaviour.

        Be aware of the volume before starting: 96 files per day, so a year is
        roughly 35,000 requests.

        Returns a summary dict.
        """
        stamps = gdelt_stamps(start, end)
        pending = [s for s in stamps
                   if overwrite or self.existing_slice(s) is None]

        summary = {"requested": len(stamps), "downloaded": 0,
                   "cached": len(stamps) - len(pending), "missing": 0,
                   "failed": 0, "events": 0}

        print(f"GDELT range {start} -> {end}: {len(stamps)} slices "
              f"(~{len(stamps) / 96:.1f} days), {summary['cached']} already local, "
              f"{len(pending)} to fetch on {max_workers} workers", flush=True)

        if not pending:
            print("Nothing to do.", flush=True)
            return summary

        lock = threading.Lock()
        started = time.perf_counter()
        done = 0

        def fetch_one(stamp):
            for attempt in range(max_retries):
                try:
                    frame = self.fetch_gdelt_hourly_export(stamp, overwrite=overwrite)
                    time.sleep(pause)
                    return stamp, (0 if frame is None else len(frame)), frame is not None
                except Exception as error:
                    if attempt == max_retries - 1:
                        return stamp, 0, None  # None marks a hard failure
                    # Exponential backoff so a transient outage does not become
                    # a tight retry loop against their server.
                    time.sleep(2 ** attempt)
            return stamp, 0, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_one, stamp): stamp for stamp in pending}
            for future in as_completed(futures):
                stamp, rows, ok = future.result()
                with lock:
                    done += 1
                    if ok is None:
                        summary["failed"] += 1
                        print(f"  {stamp}: gave up after {max_retries} attempts",
                              flush=True)
                    elif ok:
                        summary["downloaded"] += 1
                        summary["events"] += rows
                    else:
                        summary["missing"] += 1

                    if progress_every and done % progress_every == 0:
                        elapsed = time.perf_counter() - started
                        rate = done / elapsed
                        remaining = (len(pending) - done) / rate if rate else 0
                        print(f"  {done}/{len(pending)}  "
                              f"downloaded={summary['downloaded']} "
                              f"missing={summary['missing']} "
                              f"events={summary['events']:,}  "
                              f"{rate * 60:.0f}/min, ~{remaining / 60:.0f} min left",
                              flush=True)

        print(f"Done. {summary['downloaded']} downloaded, {summary['cached']} already "
              f"local, {summary['missing']} not published, {summary['failed']} failed, "
              f"{summary['events']:,} new events.", flush=True)
        return summary


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GDELT 2.0 event slices and/or market data.")
    parser.add_argument("--start", help="ISO date/time, inclusive (e.g. 2024-01-01)")
    parser.add_argument("--end", help="ISO date/time, exclusive")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--pause", type=float, default=0.05,
                        help="Seconds each worker waits between its requests")
    parser.add_argument("--workers", type=int, default=6,
                        help="Concurrent download workers (1 = sequential)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tickers", nargs="*",
                        help="Also download these tickers over the same range")
    args = parser.parse_args()

    fetcher = DataFetcher(output_dir=args.output_dir)

    if args.start and args.end:
        fetcher.fetch_gdelt_range(args.start, args.end, overwrite=args.overwrite,
                                  pause=args.pause, max_workers=args.workers)
        if args.tickers:
            # Pad the price window so returns exist on the first and last event day.
            start = (pd.Timestamp(args.start) - pd.Timedelta(days=7)).date().isoformat()
            end = (pd.Timestamp(args.end) + pd.Timedelta(days=7)).date().isoformat()
            fetcher.fetch_yfinance_data(args.tickers, start, end)
    else:
        parser.error("--start and --end are required")


if __name__ == "__main__":
    _cli()
