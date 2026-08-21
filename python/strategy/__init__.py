"""Strategy layer: instrument universe and signal construction."""

from .universe import Universe, COUNTRY_TO_ETF, COUNTRY_NAMES, THEMATIC
from .signals import (
    signal_dates,
    cutoff_timestamp,
    pairwise_pressure,
    model_signal,
    intensity_signal,
    goldstein_signal,
    event_count_signal,
    random_signal,
    reversal_signal,
)

__all__ = [
    "Universe", "COUNTRY_TO_ETF", "COUNTRY_NAMES", "THEMATIC",
    "signal_dates", "cutoff_timestamp", "pairwise_pressure", "model_signal", "intensity_signal",
    "goldstein_signal", "event_count_signal", "random_signal", "reversal_signal",
]
