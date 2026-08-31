"""System frequency excursion metrics.

Reports:
- max |Δf| from nominal
- total duration outside the operating band
"""
from __future__ import annotations

import numpy as np

from .types import MetricResult, TimeSeries

DEFAULT_NOMINAL_HZ = 60.0
DEFAULT_BAND_HZ = 0.2  # IEEE 1547 normal operating range typical bound


def frequency_excursion(
    ts: TimeSeries,
    nominal_hz: float = DEFAULT_NOMINAL_HZ,
    band_hz: float = DEFAULT_BAND_HZ,
) -> MetricResult:
    if "freq" not in ts:
        raise ValueError("time-series must contain freq column")

    t = ts["t"].to_numpy()
    f = ts["freq"].to_numpy()
    df = np.abs(f - nominal_hz)
    max_dev = float(df.max()) if len(df) else 0.0

    if len(t) > 1:
        dt = np.median(np.diff(t))
        duration_s = float((df > band_hz).sum() * dt)
    else:
        duration_s = 0.0

    return MetricResult(
        name="frequency_excursion",
        value=max_dev,
        unit="Hz",
        detail={"band_hz": band_hz, "duration_outside_band_s": duration_s},
    )
