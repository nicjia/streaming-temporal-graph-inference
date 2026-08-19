"""
Data acquisition: GDELT 2.0 event exports and Yahoo Finance price history.
"""

import io
import os
import time
import zipfile
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

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # -- Yahoo Finance ----------------------------------------------------

    def fetch_yfinance_data(self, tickers, start_date, end_date, interval="1d"):
        """Downloads historical market data for the specified tickers."""
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

        output_path = os.path.join(self.yf_dir, f"market_data_{start_date}_{end_date}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved market data to {output_path}")
        return df

    # -- GDELT ------------------------------------------------------------

    def slice_path(self, datetime_str):
        return os.path.join(self.gdelt_dir, f"gdelt_{datetime_str}.csv")

    def fetch_gdelt_hourly_export(self, datetime_str, overwrite=False, timeout=60):
        """
        Downloads and unzips one GDELT 2.0 export file.

        Format expected: YYYYMMDDHHMMSS (e.g. '20231101080000').
        Returns the parsed DataFrame, or None if the file does not exist.
        """
        output_path = self.slice_path(datetime_str)
        if os.path.exists(output_path) and not overwrite:
            return pd.read_csv(output_path, low_memory=False)

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

    def fetch_gdelt_range(self, start, end, overwrite=False, pause=0.25,
                          max_retries=3, progress_every=20):
        """
        Download every 15-minute slice in [start, end).

        Resumable: files already on disk are skipped, so re-running after an
        interruption picks up where it left off. Be aware of the volume before
        starting -- 96 files per day, a few MB each, so a month is roughly 2,900
        requests. The pause between requests is deliberate; this is someone
        else's free public server.

        Returns a summary dict.
        """
        stamps = gdelt_stamps(start, end)
        summary = {"requested": len(stamps), "downloaded": 0, "cached": 0,
                   "missing": 0, "failed": 0, "events": 0}

        print(f"GDELT range {start} -> {end}: {len(stamps)} slices "
              f"(~{len(stamps) / 96:.1f} days)")

        for index, stamp in enumerate(stamps, start=1):
            path = self.slice_path(stamp)
            if os.path.exists(path) and not overwrite:
                summary["cached"] += 1
                continue

            for attempt in range(max_retries):
                try:
                    df = self.fetch_gdelt_hourly_export(stamp, overwrite=overwrite)
                    if df is None:
                        summary["missing"] += 1
                    else:
                        summary["downloaded"] += 1
                        summary["events"] += len(df)
                    break
                except Exception as error:  # network flakiness, malformed zip
                    if attempt == max_retries - 1:
                        summary["failed"] += 1
                        print(f"  {stamp}: giving up after {max_retries} attempts ({error})")
                    else:
                        # Exponential backoff so a transient outage does not
                        # turn into a tight retry loop against their server.
                        time.sleep(2 ** attempt)

            time.sleep(pause)

            if progress_every and index % progress_every == 0:
                print(f"  {index}/{len(stamps)}  downloaded={summary['downloaded']} "
                      f"cached={summary['cached']} missing={summary['missing']} "
                      f"events={summary['events']:,}")

        print(f"Done. {summary['downloaded']} downloaded, {summary['cached']} already local, "
              f"{summary['missing']} not published, {summary['failed']} failed, "
              f"{summary['events']:,} new events.")
        return summary


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download GDELT 2.0 event slices and/or market data.")
    parser.add_argument("--start", help="ISO date/time, inclusive (e.g. 2024-01-01)")
    parser.add_argument("--end", help="ISO date/time, exclusive")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--pause", type=float, default=0.25,
                        help="Seconds between requests")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tickers", nargs="*",
                        help="Also download these tickers over the same range")
    args = parser.parse_args()

    fetcher = DataFetcher(output_dir=args.output_dir)

    if args.start and args.end:
        fetcher.fetch_gdelt_range(args.start, args.end, overwrite=args.overwrite,
                                  pause=args.pause)
        if args.tickers:
            # Pad the price window so returns exist on the first and last event day.
            start = (pd.Timestamp(args.start) - pd.Timedelta(days=7)).date().isoformat()
            end = (pd.Timestamp(args.end) + pd.Timedelta(days=7)).date().isoformat()
            fetcher.fetch_yfinance_data(args.tickers, start, end)
    else:
        parser.error("--start and --end are required")


if __name__ == "__main__":
    _cli()
