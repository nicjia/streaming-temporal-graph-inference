"""Ranking metrics for the link-prediction layer."""

import numpy as np


def roc_auc(positive_scores, negative_scores):
    """
    P(a randomly chosen positive outscores a randomly chosen negative).

    Computed from rank sums (the Mann-Whitney form) rather than by sweeping
    thresholds, and using average ranks so ties count as half a win instead of
    silently inflating the result -- an untrained model emitting near-constant
    scores would otherwise look better than chance.
    """
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    combined = np.concatenate([positive_scores, negative_scores])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)

    # Average ranks within tie groups.
    sorted_values = combined[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(positive_scores, negative_scores):
    """Area under the precision-recall curve, positives as the target class."""
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    if len(positive_scores) == 0:
        return float("nan")

    scores = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate([np.ones(len(positive_scores)),
                             np.zeros(len(negative_scores))])
    order = scores.argsort()[::-1]
    labels = labels[order]

    cumulative_hits = np.cumsum(labels)
    precision = cumulative_hits / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def mean_reciprocal_rank(positive_scores, negative_matrix):
    """
    MRR where each positive competes against its own row of negatives.

    Args:
        positive_scores: (N,)
        negative_matrix: (N, M) scores for M sampled negatives per positive.
    """
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_matrix = np.asarray(negative_matrix, dtype=np.float64)
    if positive_scores.size == 0:
        return float("nan")

    beaten_by = (negative_matrix > positive_scores[:, None]).sum(axis=1)
    return float((1.0 / (beaten_by + 1)).mean())
