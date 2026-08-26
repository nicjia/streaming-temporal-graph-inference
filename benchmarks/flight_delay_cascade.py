"""
Does the aircraft rotation graph predict delay beyond what the schedule says?

This is the cleanest case in the project for why a *temporal* graph is the right
structure, because the mechanism is physical rather than statistical. Delay
travels along the airframe. Two flights JFK->LAX an hour apart sit at the same
point in a static airport graph; their risk differs entirely according to where
each inbound aircraft currently is and how late it is running. A static graph
has no way to express that, and a per-airport time series has no way to express
which specific tail carries the delay from ORD at 14:00 to BOS at 17:00.

Absorption is also nonlinear in a way that suits continuous time: a 40-minute
scheduled turnaround swallows a 30-minute delay and the cascade stops, while a
25-minute turnaround passes it on and amplifies. The quantity that decides it is
an elapsed interval, which is what the Bochner time encoding represents.

Design:
  * Nodes are airports and tail numbers. An edge is one aircraft touching one
    airport, at departure and again at arrival, in absolute UTC.
  * Target is ArrDel15 -- the flight arrives more than 15 minutes late.
  * Everything is read at scheduled departure minus 60 minutes. The sampler
    only returns edges strictly before the query time, so no model can see the
    inbound aircraft land if it had not landed by then.

Three models share one MLP head, one optimiser and one training budget, so the
only thing that varies between them is what they are allowed to look at:
  A  schedule only          -- hour, weekday, distance, carrier, airports
  B  schedule + rotation    -- adds the hand-built inbound-delay and slack
                               features, the standard way this is done
  C  schedule + TGAT        -- adds learned embeddings of the tail and origin

B is the interesting comparison. Hand-engineered rotation features are what the
published literature uses to capture propagation (FlightSense, arXiv 2605.07364,
reports ROC AUC 0.732 from schedule alone rising to 0.875 once rotation-chain
features are added). If C cannot beat B, the network is not earning its keep.

Usage:  python benchmarks/flight_delay_cascade.py --months 6
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))

import graph_engine  # noqa: E402
from backtest.evaluate import roc_auc  # noqa: E402
from ingestion.flights import build_events, load, solve_offsets  # noqa: E402
from models import PCSRTemporalSampler, TGAT  # noqa: E402

LEAD = 3600            # everything is read this many seconds before departure

# Timing alone does not encode lateness. An arrival at 14:37 is early or two
# hours late depending on a schedule the graph does not contain, so a model
# reading only event times cannot recover the one quantity that propagates.
# The engine carries a uint16 relation per edge; bucketing the delay into it
# gives the network the same information the hand-built features read off
# ArrDelay directly, and keeps the comparison honest.
DELAY_EDGES = np.array([-1e9, 0.0, 15.0, 45.0, 1e9])
N_BUCKET = len(DELAY_EDGES) - 1


def delay_relation(minutes, arriving):
    bucket = np.clip(np.digitize(minutes, DELAY_EDGES[1:-1]), 0, N_BUCKET - 1)
    return (bucket + (N_BUCKET if arriving else 0)).astype(np.uint16)


def rotation_features(events):
    """Inbound-aircraft state, restricted to what is known one hour out.

    This is the hand-built version of the propagation channel: how late the
    aircraft's previous leg landed, and how much scheduled slack sits between
    that landing and this departure. The causal cut matters more than the
    features -- a previous leg that has not landed by the cutoff is unknown,
    and is marked missing rather than filled with the answer.
    """
    order = np.lexsort((events["sched_dep"].to_numpy(),
                        events["Tail_Number"].to_numpy()))
    ordered = events.iloc[order]
    same_tail = (ordered["Tail_Number"].to_numpy()[1:]
                 == ordered["Tail_Number"].to_numpy()[:-1])
    same_tail = np.r_[False, same_tail]

    prev_arr = np.r_[0, ordered["actual_arr"].to_numpy()[:-1]]
    prev_delay = np.r_[0.0, ordered["ArrDelay"].to_numpy()[:-1]]
    dep = ordered["sched_dep"].to_numpy()

    known = same_tail & (prev_arr < dep - LEAD)
    slack = np.where(known, (dep - prev_arr) / 3600.0, 0.0)
    inbound = np.where(known, prev_delay, 0.0)

    out = pd.DataFrame({"inbound_delay": inbound, "slack_hours": slack,
                        "inbound_known": known.astype(np.float64)},
                       index=ordered.index)
    return out.reindex(events.index)


def design_matrix(events, extra=None):
    """Dense schedule features. Categoricals are returned separately as codes.

    One-hot encoding 335 origins, 335 destinations and the carrier across two
    million flights would allocate roughly six gigabytes per matrix, three
    times over. Embedding layers carry the same information in a few hundred
    kilobytes and let the categories share statistical strength.
    """
    hour = events["dep_min"].to_numpy() / 60.0
    parts = [
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        events["date"].dt.dayofweek.to_numpy() / 6.0,
        events["date"].dt.month.to_numpy() / 12.0,
        np.log1p(events["Distance"].to_numpy()) / 10.0,
        events["CRSElapsedTime"].to_numpy() / 600.0,
    ]
    matrix = np.stack(parts, axis=1).astype(np.float32)
    if extra is not None:
        matrix = np.hstack([matrix, extra.to_numpy().astype(np.float32)])
    return matrix


def category_codes(events):
    codes = [pd.Categorical(events[c]).codes.astype(np.int64)
             for c in ("Reporting_Airline", "Origin", "Dest")]
    sizes = [int(c.max()) + 1 for c in codes]
    return np.stack(codes, axis=1), sizes


class Head(nn.Module):
    """One architecture for all three models; only the input width changes."""

    def __init__(self, dense_dim, cat_sizes, emb=16, hidden=128):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, emb) for n in cat_sizes])
        width = dense_dim + emb * len(cat_sizes)
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, hidden // 2),
                                 nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, dense, cats):
        parts = [dense] + [e(cats[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


def train_head(x_tr, c_tr, y_tr, x_te, c_te, y_te, cat_sizes, epochs, batch,
               seed=0):
    torch.manual_seed(seed)
    model = Head(x_tr.shape[1], cat_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(x_tr)
    ct = torch.from_numpy(c_tr)
    yt = torch.from_numpy(y_tr.astype(np.float32))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for off in range(0, len(xt), batch):
            idx = perm[off:off + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx], ct[idx]), yt[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    scores = np.empty(len(x_te), dtype=np.float64)
    with torch.no_grad():
        for off in range(0, len(x_te), 8192):
            end = min(off + 8192, len(x_te))
            scores[off:end] = model(torch.from_numpy(x_te[off:end]),
                                    torch.from_numpy(c_te[off:end])).numpy()
    return roc_auc(scores[y_te == 1], scores[y_te == 0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=6)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--tgat-epochs", type=int, default=3)
    p.add_argument("--tgat-events", type=int, default=120000)
    p.add_argument("--node-dim", type=int, default=64)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-embed", type=int, default=120000)
    args = p.parse_args()

    frame = load(os.path.join(ROOT, "data/bts"), limit=args.months)
    offsets = solve_offsets(frame)
    events = build_events(frame, offsets)
    print(f"flights: {len(events):,}  tails: {events['Tail_Number'].nunique():,}  "
          f"airports: {events['Origin'].nunique():,}")
    print(f"ArrDel15 base rate: {events['ArrDel15'].mean():.1%}")

    # ---- temporal graph: tail <-> airport, departures and arrivals ---------
    tails = pd.Categorical(events["Tail_Number"])
    airports = pd.Categorical(pd.concat([events["Origin"], events["Dest"]]))
    n_air = len(airports.categories)
    air_code = {a: i for i, a in enumerate(airports.categories)}
    tail_id = tails.codes.astype(np.int64) + n_air
    origin_id = events["Origin"].map(air_code).to_numpy(np.int64)
    dest_id = events["Dest"].map(air_code).to_numpy(np.int64)
    vertices = n_air + len(tails.categories)

    actual_dep = (events["sched_dep"].to_numpy()
                  + events["DepDelay"].to_numpy() * 60).astype(np.int64)
    actual_arr = events["actual_arr"].to_numpy()

    src = np.concatenate([tail_id, origin_id, tail_id, dest_id])
    dst = np.concatenate([origin_id, tail_id, dest_id, tail_id])
    ts = np.concatenate([actual_dep, actual_dep, actual_arr, actual_arr])
    dep_rel = delay_relation(events["DepDelay"].to_numpy(), arriving=False)
    arr_rel = delay_relation(events["ArrDelay"].to_numpy(), arriving=True)
    rel = np.concatenate([dep_rel, dep_rel, arr_rel, arr_rel])
    order = np.argsort(ts, kind="stable")
    src, dst, ts, rel = src[order], dst[order], ts[order], rel[order]
    base = ts.min()
    ts = (ts - base).astype(np.uint32)

    graph = graph_engine.PCSRGraph(vertices, int(len(src) * 2.2),
                                   int(len(src) * 2.2) * 8 * 4 + (1 << 26))
    graph.insert_edges(src.astype(np.uint32), dst.astype(np.uint32), ts, rel)
    sampler = PCSRTemporalSampler(graph)
    print(f"graph: {graph.num_edges:,} edges over {vertices:,} nodes "
          f"({n_air} airports + {len(tails.categories):,} tails), "
          f"chronological: {sampler.validate()}")
    counts = np.bincount(rel, minlength=2 * N_BUCKET)
    print("edge relations (delay bucket): "
          + "  ".join(f"{'arr' if i >= N_BUCKET else 'dep'}"
                      f"{['<0','0-15','15-45','45+'][i % N_BUCKET]}={c:,}"
                      for i, c in enumerate(counts) if c))

    # ---- chronological split ----------------------------------------------
    cut = int(len(events) * 0.75)
    y = events["ArrDel15"].to_numpy().astype(np.int64)
    print(f"split: train {cut:,} | test {len(events)-cut:,}  "
          f"(test begins {events['date'].iloc[cut].date()})")

    rot = rotation_features(events)
    print(f"inbound aircraft known one hour out for {rot['inbound_known'].mean():.1%} "
          f"of flights")

    x_sched = design_matrix(events)
    x_rot = design_matrix(events, rot)
    cats, cat_sizes = category_codes(events)
    print(f"features: {x_sched.shape[1]} dense + {len(cat_sizes)} embedded "
          f"categoricals {cat_sizes}")

    print(f"\n{'model':<34}{'inputs':>9}{'AUC':>9}{'minutes':>10}")
    print("-" * 62)
    started = time.perf_counter()
    auc_a = train_head(x_sched[:cut], cats[:cut], y[:cut],
                       x_sched[cut:], cats[cut:], y[cut:],
                       cat_sizes, args.epochs, args.batch)
    print(f"{'A  schedule only':<34}{x_sched.shape[1]+48:>9}{auc_a:>9.4f}"
          f"{(time.perf_counter()-started)/60:>10.1f}", flush=True)

    started = time.perf_counter()
    auc_b = train_head(x_rot[:cut], cats[:cut], y[:cut],
                       x_rot[cut:], cats[cut:], y[cut:],
                       cat_sizes, args.epochs, args.batch)
    print(f"{'B  + hand-built rotation':<34}{x_rot.shape[1]+48:>9}{auc_b:>9.4f}"
          f"{(time.perf_counter()-started)/60:>10.1f}", flush=True)

    # ---- C: TGAT embeddings of the tail and origin, one hour out ----------
    started = time.perf_counter()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tgat = TGAT(vertices, node_dim=args.node_dim, time_dim=args.node_dim,
                num_layers=args.layers, num_neighbors=args.neighbors,
                sampler=sampler, num_relations=2 * N_BUCKET)
    label = torch.from_numpy(y.astype(np.float32))
    probe = ((events["sched_dep"].to_numpy() - base) - LEAD).clip(0).astype(np.uint32)

    # Train the encoder jointly with a thin readout on the training window.
    readout = nn.Linear(args.node_dim * 2, 1)
    params = list(tgat.parameters()) + list(readout.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    step = max(1, cut // args.tgat_events)
    idx_train = np.arange(0, cut, step)
    tgat.train()
    for epoch in range(args.tgat_epochs):
        losses = []
        for off in range(0, len(idx_train), 128):
            part = idx_train[off:off + 128]
            if len(part) < 8:
                continue
            optimizer.zero_grad()
            h_tail = tgat(tail_id[part], probe[part])
            h_air = tgat(origin_id[part], probe[part])
            logit = readout(torch.cat([h_tail, h_air], dim=1)).squeeze(-1)
            loss = loss_fn(logit, label[part])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"   tgat epoch {epoch+1}: loss {np.mean(losses):.4f}", flush=True)

    # Freeze and emit embeddings as features for the shared head.
    tgat.eval()
    keep = np.arange(len(events))
    if len(keep) > args.max_embed:
        keep = np.unique(np.r_[
            np.linspace(0, cut - 1, args.max_embed // 2).astype(int),
            np.linspace(cut, len(events) - 1, args.max_embed // 2).astype(int)])
    embeds = np.zeros((len(events), args.node_dim * 2), dtype=np.float32)
    with torch.no_grad():
        for off in range(0, len(keep), 2048):
            part = keep[off:off + 2048]
            h_tail = tgat(tail_id[part], probe[part]).numpy()
            h_air = tgat(origin_id[part], probe[part]).numpy()
            embeds[part] = np.hstack([h_tail, h_air])
    sub = keep
    sub_cut = int(np.searchsorted(sub, cut))
    x_tgat = np.hstack([x_sched, embeds])
    auc_c = train_head(x_tgat[sub[:sub_cut]], cats[sub[:sub_cut]], y[sub[:sub_cut]],
                       x_tgat[sub[sub_cut:]], cats[sub[sub_cut:]], y[sub[sub_cut:]],
                       cat_sizes, args.epochs, args.batch)
    print(f"{'C  + TGAT rotation embedding':<34}{x_tgat.shape[1]+48:>9}{auc_c:>9.4f}"
          f"{(time.perf_counter()-started)/60:>10.1f}", flush=True)

    print(f"\nlift from hand-built rotation features : {auc_b-auc_a:+.4f}")
    print(f"lift from learned rotation embedding   : {auc_c-auc_a:+.4f}")
    print("\nPublished reference (FlightSense, arXiv 2605.07364, XGBoost on 7.07M")
    print("BTS 2018 records): 0.732 schedule-only -> 0.875 with hand-engineered")
    print("aircraft-rotation-chain features. Different model class and sample, so")
    print("the internal A/B/C comparison above is the controlled one.")


if __name__ == "__main__":
    main()
