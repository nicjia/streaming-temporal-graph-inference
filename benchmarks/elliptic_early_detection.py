"""Causal versus non-causal graph information in Elliptic AML detection.

Every Elliptic edge joins transactions in the same coarse two-week timestep,
so this dataset cannot test a TGAT's cross-time memory.  It can still answer a
useful early-detection question: do *predecessor* transactions, whose outputs
fund the focal transaction, improve illicit classification beyond its local
features?  Outgoing children are excluded because they do not exist yet when
the focal transaction arrives.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(root):
    feature_path = os.path.join(root, "elliptic_txs_features.csv")
    # 167 columns: id, timestep, 93 local transaction features and 72
    # precomputed aggregate features. float32 halves the 690 MB CSV footprint.
    dtype = {0: np.int64, 1: np.int16}
    dtype.update({column: np.float32 for column in range(2, 167)})
    raw = pd.read_csv(feature_path, header=None, dtype=dtype)
    ids = raw.iloc[:, 0].to_numpy(dtype=np.int64)
    timestep = raw.iloc[:, 1].to_numpy(dtype=np.int16)
    local = raw.iloc[:, 2:95].to_numpy(dtype=np.float32)
    published_aggregate = raw.iloc[:, 95:167].to_numpy(dtype=np.float32)
    classes = pd.read_csv(os.path.join(root, "elliptic_txs_classes.csv"))
    class_map = dict(zip(classes["txId"].astype(np.int64), classes["class"].astype(str)))
    label = np.array([class_map.get(tx, "unknown") for tx in ids])
    edges = pd.read_csv(os.path.join(root, "elliptic_txs_edgelist.csv"))
    index = {tx: i for i, tx in enumerate(ids)}
    source = edges.iloc[:, 0].map(index).to_numpy(dtype=np.int64)
    target = edges.iloc[:, 1].map(index).to_numpy(dtype=np.int64)
    return ids, timestep, local, published_aggregate, label, source, target


def predecessor_features(local, source, target, timestep, shuffle=False, seed=0):
    source = source.copy()
    if shuffle:
        # Keep each target and its indegree fixed; permute source endpoints only
        # inside the same timestep so no edge crosses time or leaks the future.
        rng = np.random.default_rng(seed)
        edge_step = timestep[target]
        for step in np.unique(edge_step):
            rows = np.flatnonzero(edge_step == step)
            source[rows] = rng.permutation(source[rows])
    degree = np.bincount(target, minlength=len(local)).astype(np.float32)
    aggregate = np.zeros_like(local, dtype=np.float32)
    np.add.at(aggregate, target, local[source])
    aggregate /= np.maximum(degree[:, None], 1.0)
    return np.column_stack([np.log1p(degree), aggregate]).astype(np.float32)


def fit_model(X, y, train, seed=0):
    model = HistGradientBoostingClassifier(
        max_iter=250, max_leaf_nodes=31, learning_rate=.05,
        l2_regularization=2, random_state=seed)
    positive_weight = max((~y[train]).sum() / max(y[train].sum(), 1), 1)
    weight = np.where(y[train], positive_weight, 1.0)
    model.fit(X[train], y[train], sample_weight=weight)
    return model


def select_threshold(target, score):
    candidates = np.quantile(score, np.linspace(.70, .995, 200))
    f1 = np.array([f1_score(target, score >= value) for value in candidates])
    return float(candidates[np.argmax(f1)])


def report(name, model, X, y, validation, test):
    validation_score = model.predict_proba(X[validation])[:, 1]
    score = model.predict_proba(X[test])[:, 1]
    threshold = select_threshold(y[validation], validation_score)
    prediction = score >= threshold
    result = {
        "auc": roc_auc_score(y[test], score),
        "ap": average_precision_score(y[test], score),
        "f1": f1_score(y[test], prediction),
        "precision": precision_score(y[test], prediction, zero_division=0),
        "recall": recall_score(y[test], prediction, zero_division=0),
    }
    print(f"{name:<28} AUC {result['auc']:.3f} | AP {result['ap']:.3f} | "
          f"F1 {result['f1']:.3f} | P/R {result['precision']:.3f}/{result['recall']:.3f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "data/elliptic"))
    args = parser.parse_args()
    ids, step, local, published, raw_label, source, target = load_data(args.data)
    labeled = np.isin(raw_label, ["1", "2"])
    y = raw_label == "1"
    train = labeled & (step <= 34)
    validation = labeled & (step >= 35) & (step <= 39)
    test = labeled & (step >= 40)
    same_step = np.mean(step[source] == step[target])
    print(f"Elliptic: {len(ids):,} transactions | {len(source):,} edges | "
          f"{labeled.sum():,} labeled | {same_step:.1%} edges within one timestep")
    print(f"chronological split: train {train.sum():,} | validation {validation.sum():,} | "
          f"test {test.sum():,} | test illicit {y[test].mean():.1%}\n")

    causal = predecessor_features(local, source, target, step)
    shuffled = predecessor_features(local, source, target, step, shuffle=True)
    variants = {
        "local transaction only": local,
        "local + causal predecessors": np.column_stack([local, causal]),
        "local + shuffled predecessors": np.column_stack([local, shuffled]),
        "local + published aggregate": np.column_stack([local, published]),
    }
    for name, X in variants.items():
        model = fit_model(X, y, train)
        report(name, model, X, y, validation, test)


if __name__ == "__main__":
    main()

