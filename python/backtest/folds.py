"""
Walk-forward fold construction.

The whole point of walk-forward is that the model deciding a trade on day d was
fitted only on data that existed before day d. Any harness that fits once on
everything and then "backtests" over the same span is measuring memorisation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int   # unix seconds, inclusive
    train_end: int     # unix seconds, exclusive -- also the test start
    test_end: int      # unix seconds, exclusive

    @property
    def test_start(self):
        return self.train_end

    def describe(self):
        def fmt(ts):
            return pd.Timestamp(ts, unit="s").date().isoformat()
        return (f"fold {self.index}: train {fmt(self.train_start)}..{fmt(self.train_end)} "
                f"| test {fmt(self.train_end)}..{fmt(self.test_end)}")


def walk_forward_folds(timestamps, num_folds=4, min_train_fraction=0.4,
                       expanding=True):
    """
    Split a timeline into sequential train/test folds.

    Args:
        timestamps: Event timestamps (unix seconds); only min and max are used.
        num_folds: Number of test windows.
        min_train_fraction: Fraction of the span reserved as the initial
            training period before the first test window opens.
        expanding: True keeps train_start pinned at the beginning so each fold
            trains on everything available so far -- the honest default, since
            in production you would not throw old data away. False gives a
            rolling window of constant length, useful for probing whether the
            relationship is stable.

    Returns:
        list[Fold]
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if timestamps.size == 0:
        raise ValueError("no timestamps to split")

    start, end = int(timestamps.min()), int(timestamps.max()) + 1
    span = end - start
    if span <= 0:
        raise ValueError("timeline has zero span")

    first_test = start + int(span * min_train_fraction)
    test_span = (end - first_test) / num_folds
    if test_span <= 0:
        raise ValueError("min_train_fraction leaves no room for test windows")

    folds = []
    for index in range(num_folds):
        train_end = int(first_test + index * test_span)
        test_end = int(first_test + (index + 1) * test_span) if index < num_folds - 1 else end
        train_start = start if expanding else max(start, train_end - (first_test - start))
        folds.append(Fold(index, train_start, train_end, test_end))
    return folds
