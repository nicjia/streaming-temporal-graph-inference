"""Temporal graph models over the C++ PCSR engine."""

from .temporal_encoding import TimeEncode
from .neighbor_sampler import PCSRTemporalSampler
from .tgat import TemporalAttentionLayer, TGAT, LinkPredictor, TGATLinkModel
from .forecast import (
    ReturnForecastModel,
    build_return_targets,
    train_forecast,
    forecast_signal,
    information_coefficient,
)
from .intensity import (
    IntensityHead,
    TGATIntensityModel,
    build_intensity_targets,
    train_intensity,
)

__all__ = [
    "TimeEncode",
    "PCSRTemporalSampler",
    "TemporalAttentionLayer",
    "TGAT",
    "LinkPredictor",
    "TGATLinkModel",
    "IntensityHead",
    "TGATIntensityModel",
    "build_intensity_targets",
    "train_intensity",
    "ReturnForecastModel",
    "build_return_targets",
    "train_forecast",
    "forecast_signal",
    "information_coefficient",
]
