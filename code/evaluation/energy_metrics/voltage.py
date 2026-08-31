"""Voltage deviation metrics per ANSI C84.1 / IEEE 1547 service ranges.

Reports:
- fraction of (bus, time) samples outside the band
- worst per-bus excursion magnitude
"""
from __future__ import annotations

import numpy as np

from .types import MetricResult, TimeSeries

DEFAULT_BAND_PU = (0.95, 1.05)  # ANSI C84.1 Range A on a normalized basis


def voltage_deviation(ts: TimeSeries, band_pu: tuple[float, float] = DEFAULT_BAND_PU) -> MetricResult:
    cols = [c for c in ts.columns if c.startswith("v_pu_")]
    if not cols:
        raise ValueError("no v_pu_<bus> columns found")

    v = ts[cols].to_numpy()
    lo, hi = band_pu
    out_of_band = (v < lo) | (v > hi)
    frac = float(out_of_band.mean())

    worst_per_bus = {
        cols[i].removeprefix("v_pu_"): float(np.abs(v[:, i] - 1.0).max())
        for i in range(v.shape[1])
    }
    return MetricResult(
        name="voltage_deviation",
        value=frac,
        unit="fraction",
        detail={"band_pu": list(band_pu), "worst_per_bus_pu": worst_per_bus},
    )
