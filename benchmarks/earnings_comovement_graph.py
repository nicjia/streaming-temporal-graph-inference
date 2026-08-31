"""Point-in-time latent return graph for earnings-tail prediction."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from models.earnings_distribution import build_pre_event_features


def paired_date_bootstrap(frame, left, right, repetitions=1000, seed=0):
    dates = pd.to_datetime(frame["anndats_act"]).dt.strftime("%Y-%m-%d").to_numpy()
    unique = np.unique(dates)
    rows = {date: np.flatnonzero(dates == date) for date in unique}
    target = frame["target"].to_numpy()
    rng = np.random.default_rng(seed)
    auc, ap = [], []
    for _ in range(repetitions):
        index = np.concatenate([rows[date] for date in rng.choice(
            unique, len(unique), replace=True)])
        truth = target[index]
        auc.append(roc_auc_score(truth, left[index])
                   - roc_auc_score(truth, right[index]))
        ap.append(average_precision_score(truth, left[index])
                  - average_precision_score(truth, right[index]))
    return np.quantile(auc, [.025, .5, .975]), np.quantile(ap, [.025, .5, .975])


def build_annual_neighbors(events, returns, components=8, neighbors=10, seed=0):
    """Fit each year's graph using only daily returns from the previous year."""
    graphs = {}
    for year in range(max(2016, events["anndats_act"].dt.year.min()),
                      events["anndats_act"].dt.year.max() + 1):
        universe = events.loc[events["anndats_act"].dt.year.eq(year),
                              "permno"].unique()
        prior = returns[(returns["date"].dt.year == year - 1)
                        & returns["permno"].isin(universe)]
        coverage = prior.groupby("permno")["ret"].count()
        usable = coverage[coverage >= 100].index
        prior = prior[prior["permno"].isin(usable)]
        matrix = prior.pivot(index="permno", columns="date", values="ret")
        matrix = matrix.clip(-.25, .25)
        values = matrix.to_numpy(dtype=float)
        row_mean = np.nanmean(values, axis=1, keepdims=True)
        row_std = np.nanstd(values, axis=1, keepdims=True)
        values = (values - row_mean) / np.where(row_std > 1e-8, row_std, 1)
        values = np.nan_to_num(values)
        dimension = min(components, values.shape[0] - 1, values.shape[1] - 1)
        if dimension < 2:
            continue
        embedding = PCA(n_components=dimension, svd_solver="randomized",
                        random_state=seed).fit_transform(values)
        count = min(neighbors + 1, len(matrix))
        _, indices = NearestNeighbors(n_neighbors=count, metric="euclidean").fit(
            embedding).kneighbors(embedding)
        permnos = matrix.index.to_numpy(dtype=int)
        graphs[year] = {
            int(permno): permnos[row[1:]].astype(int)
            for permno, row in zip(permnos, indices)
        }
        print(f"{year}: {len(permnos):,} nodes, {sum(map(len, graphs[year].values())):,} "
              "directed latent edges", flush=True)
    return graphs


def aggregate_peer_events(frame, graphs, randomize=False, seed=0):
    ordered = frame.sort_values(["reaction_date", "event_id"], kind="stable")
    histories = {}
    for permno, group in ordered.groupby("permno", sort=False):
        histories[int(permno)] = {
            "date": group["reaction_date"].to_numpy(dtype="datetime64[D]"),
            "abs": group["abs_reaction_1d"].to_numpy(dtype=float),
            "tail": group["tail_15"].to_numpy(dtype=float),
            "signed": group["reaction_1d"].to_numpy(dtype=float),
            "surprise": group["eps_surprise_z"].to_numpy(dtype=float),
        }
    rng = np.random.default_rng(seed)
    rows = []
    for event in frame.itertuples(index=False):
        year = pd.Timestamp(event.anndats_act).year
        graph = graphs.get(year, {})
        peers = graph.get(int(event.permno), np.array([], dtype=int))
        if randomize and len(peers) and len(graph) > len(peers):
            universe = np.fromiter(graph.keys(), dtype=int)
            universe = universe[universe != int(event.permno)]
            peers = rng.choice(universe, len(peers), replace=False)
        date = np.datetime64(event.anndats_act, "D")
        item = {"event_id": int(event.event_id), "latent_degree": len(peers)}
        for days in (30, 90, 365):
            aggregate = {name: [] for name in ("abs", "tail", "signed", "surprise")}
            start = date - np.timedelta64(days, "D")
            for peer in peers:
                history = histories.get(int(peer))
                if history is None:
                    continue
                left = np.searchsorted(history["date"], start, side="left")
                right = np.searchsorted(history["date"], date, side="left")
                for name in aggregate:
                    aggregate[name].extend(history[name][left:right])
            item[f"latent_events_{days}d"] = len(aggregate["abs"])
            for name, values in aggregate.items():
                item[f"latent_{name}_{days}d"] = (
                    float(np.mean(values)) if values else np.nan)
        rows.append(item)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=os.path.join(
        ROOT, "data/earnings/reactions.parquet"))
    parser.add_argument("--returns", nargs="+", default=[
        os.path.join(ROOT, "data/crsp_dsf.parquet"),
        os.path.join(ROOT, "data/crsp_dsf_extra.parquet")])
    parser.add_argument("--features", default=os.path.join(
        ROOT, "data/earnings/comovement_features.parquet"))
    args = parser.parse_args()

    frame, base_features = build_pre_event_features(pd.read_parquet(args.events))
    if os.path.exists(args.features):
        latent = pd.read_parquet(args.features)
    else:
        pieces = [pd.read_parquet(path, columns=["permno", "date", "ret"])
                  for path in args.returns]
        returns = pd.concat(pieces, ignore_index=True)
        returns["date"] = pd.to_datetime(returns["date"])
        returns["ret"] = pd.to_numeric(returns["ret"], errors="coerce")
        returns = returns.dropna(subset=["ret"]).drop_duplicates(
            ["permno", "date"], keep="first")
        graphs = build_annual_neighbors(frame, returns)
        actual = aggregate_peer_events(frame, graphs)
        random = aggregate_peer_events(frame, graphs, randomize=True).rename(
            columns=lambda x: x if x == "event_id" else "random_" + x)
        latent = actual.merge(random, on="event_id", validate="one_to_one")
        os.makedirs(os.path.dirname(args.features), exist_ok=True)
        latent.to_parquet(args.features, index=False)
    frame = frame.merge(latent, on="event_id", validate="one_to_one")
    train = frame[(frame["anndats_act"] >= "2016-01-01")
                  & (frame["anndats_act"] <= "2021-12-31")].copy()
    validation = frame[(frame["anndats_act"] > "2021-12-31")
                       & (frame["anndats_act"] <= "2022-12-31")].copy()
    test = frame[frame["anndats_act"] > "2022-12-31"].copy().reset_index(drop=True)
    graph_features = [column for column in frame if column.startswith("latent_")]
    random_features = [column for column in frame if column.startswith("random_latent_")]
    for split in (train, validation, test):
        split.loc[:, "target"] = split["tail_15"]
    predictions = {}
    for name, features in (
        ("non-graph", base_features),
        ("non-graph + latent", base_features + graph_features),
        ("non-graph + random", base_features + random_features),
    ):
        model = HistGradientBoostingClassifier(
            max_iter=250, max_leaf_nodes=31, learning_rate=.05,
            l2_regularization=2, random_state=0)
        model.fit(train[features], train["target"])
        score = model.predict_proba(test[features])[:, 1]
        predictions[name] = score
        print(f"{name:<24} AUC {roc_auc_score(test['target'], score):.3f} | "
              f"AP {average_precision_score(test['target'], score):.3f}")
    for challenger in ("non-graph + latent", "non-graph + random"):
        auc, ap = paired_date_bootstrap(
            test, predictions[challenger], predictions["non-graph"])
        print(f"{challenger} minus non-graph: AUC "
              f"[{auc[0]:+.4f},{auc[2]:+.4f}] median {auc[1]:+.4f} | "
              f"AP [{ap[0]:+.4f},{ap[2]:+.4f}] median {ap[1]:+.4f}")
    auc, ap = paired_date_bootstrap(
        test, predictions["non-graph + latent"], predictions["non-graph + random"])
    print(f"latent minus random peers: AUC [{auc[0]:+.4f},{auc[2]:+.4f}] "
          f"median {auc[1]:+.4f} | AP [{ap[0]:+.4f},{ap[2]:+.4f}] "
          f"median {ap[1]:+.4f}")


if __name__ == "__main__":
    main()
