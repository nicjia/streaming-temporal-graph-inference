"""
BTS on-time performance as a temporal aircraft-rotation graph.

Delay propagates along the physical aircraft, not the route map. Two flights
JFK->LAX an hour apart occupy the same position in a static airport graph and
carry entirely different risk depending on where each inbound airframe is and
how late it is running. That dependency is only visible if the tail number is a
node and the edges are ordered in absolute time.

Absolute time is the catch. BTS reports every clock field in LOCAL time and
ships no timezone column, so a naive parse interleaves an 08:00 departure from
Los Angeles with an 08:00 departure from New York and the rotation chain comes
out scrambled. The offsets are recoverable from the data itself: for any leg,

    local_arrival - local_departure = elapsed + (offset_dest - offset_origin)

and CRSElapsedTime is reported, so every flight observes one pairwise
difference. Anchoring one airport and relaxing over the route graph solves the
rest. No external timezone table, and the result is checkable -- offsets should
come out as whole or half hours.
"""

import glob
import os
import zipfile

import numpy as np
import pandas as pd

COLUMNS = ["FlightDate", "Tail_Number", "Origin", "Dest", "CRSDepTime",
           "DepDelay", "CRSArrTime", "ArrDelay", "CRSElapsedTime", "Distance",
           "Reporting_Airline", "Cancelled", "Diverted", "ArrDel15",
           "LateAircraftDelay"]


def _hhmm_to_minutes(series):
    value = pd.to_numeric(series, errors="coerce")
    return (value // 100) * 60 + (value % 100)


def load_month(path):
    with zipfile.ZipFile(path) as archive:
        name = [n for n in archive.namelist() if n.endswith(".csv")][0]
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, usecols=COLUMNS, low_memory=False)
    return frame


def load(directory, limit=None):
    # Sort by month number, not by filename. Lexicographic order puts
    # bts_2024_10 immediately after bts_2024_1, so asking for the first four
    # months silently returns January, October, November and December -- four
    # months with three gaps, which severs every rotation chain between them.
    def month_of(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        tail = stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    files = sorted(glob.glob(os.path.join(directory, "*.zip")), key=month_of)
    # A partially written archive is indistinguishable from a corrupt one, and
    # both are ordinary when downloads run alongside analysis. Skip and say so
    # rather than failing the whole run on one bad month.
    usable, skipped = [], []
    for path in files:
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is None:
                    usable.append(path)
                    continue
        except (zipfile.BadZipFile, OSError):
            pass
        skipped.append(os.path.basename(path))
    if skipped:
        print(f"skipping {len(skipped)} unreadable archive(s): {', '.join(skipped)}")
    if limit:
        usable = usable[:limit]
    if not usable:
        raise SystemExit(f"no readable BTS zips in {directory}")
    frame = pd.concat([load_month(f) for f in usable], ignore_index=True)

    frame = frame[(frame["Cancelled"] == 0) & (frame["Diverted"] == 0)]
    frame = frame.dropna(subset=["Tail_Number", "ArrDelay", "DepDelay",
                                 "CRSElapsedTime", "ArrDel15"])
    frame["dep_min"] = _hhmm_to_minutes(frame["CRSDepTime"])
    frame["arr_min"] = _hhmm_to_minutes(frame["CRSArrTime"])
    frame["date"] = pd.to_datetime(frame["FlightDate"])
    return frame.reset_index(drop=True)


def solve_offsets(frame, anchor="ATL", rounds=40):
    """Recover each airport's UTC offset in minutes from scheduled times."""
    diff = (frame["arr_min"] - frame["dep_min"]).to_numpy()
    # A scheduled arrival before its departure has crossed midnight.
    diff = np.where(diff < -720, diff + 1440, np.where(diff > 720, diff - 1440, diff))
    implied = diff - frame["CRSElapsedTime"].to_numpy()

    edge = (pd.DataFrame({"o": frame["Origin"].to_numpy(),
                          "d": frame["Dest"].to_numpy(),
                          "delta": implied})
            .groupby(["o", "d"])["delta"].median().reset_index())

    airports = sorted(set(edge["o"]) | set(edge["d"]))
    offset = {a: 0.0 for a in airports}
    if anchor not in offset:
        anchor = airports[0]

    # Relaxation: offset[d] = offset[o] + delta over every observed route,
    # re-anchored each round so the solution cannot drift as a whole.
    into = {k: v for k, v in edge.groupby("d")}
    outof = {k: v for k, v in edge.groupby("o")}
    for _ in range(rounds):
        update = {}
        for airport in airports:
            estimates = []
            part = into.get(airport)
            if part is not None:
                estimates.extend(offset[o] + dl
                                 for o, dl in zip(part["o"], part["delta"]))
            part = outof.get(airport)
            if part is not None:
                estimates.extend(offset[d] - dl
                                 for d, dl in zip(part["d"], part["delta"]))
            if estimates:
                update[airport] = float(np.median(estimates))
        base = update.get(anchor, 0.0)
        offset = {a: v - base for a, v in update.items()}
    return offset


def build_events(frame, offset):
    """Absolute scheduled departure / actual arrival, in unix seconds."""
    off_o = frame["Origin"].map(offset).astype("float64")
    off_d = frame["Dest"].map(offset).astype("float64")
    keep = off_o.notna() & off_d.notna()
    frame = frame[keep].copy()
    off_o, off_d = off_o[keep], off_d[keep]

    epoch = (frame["date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    frame["sched_dep"] = (epoch + (frame["dep_min"] - off_o) * 60).astype("int64")
    frame["sched_arr"] = (frame["sched_dep"]
                          + frame["CRSElapsedTime"].astype("float64") * 60).astype("int64")
    frame["actual_arr"] = (frame["sched_arr"]
                           + frame["ArrDelay"].astype("float64") * 60).astype("int64")
    return frame.sort_values("sched_dep").reset_index(drop=True)
