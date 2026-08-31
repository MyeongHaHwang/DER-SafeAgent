"""Energy Not Supplied (ENS).

ENS [kWh] = integral over the event window of max(0, demand - served).
"""
from __future__ import annotations

import numpy as np

from .types import MetricResult, TimeSeries


def energy_not_supplied(ts: TimeSeries) -> MetricResult:
    if "load_demand" not in ts or "load_served" not in ts:
        raise ValueError("time-series must contain load_demand and load_served")

    t = ts["t"].to_numpy()
    deficit_kw = np.clip(ts["load_demand"].to_numpy() - ts["load_served"].to_numpy(), 0.0, None)
    # trapezoidal integration; t in seconds → divide by 3600 for kWh
    ens_kwh = float(np.trapezoid(deficit_kw, t) / 3600.0)

    peak_deficit = float(deficit_kw.max()) if len(deficit_kw) else 0.0
    duration_s = float((deficit_kw > 1e-6).sum() * np.median(np.diff(t))) if len(t) > 1 else 0.0

    return MetricResult(
        name="ens",
        value=ens_kwh,
        unit="kWh",
        detail={"peak_deficit_kw": peak_deficit, "deficit_duration_s": duration_s},
    )
