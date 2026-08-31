"""Curtailed energy — proxy for false-positive cost of overzealous mitigations.

Curtailment [kWh] = integral over event window of max(0, P_avail - P_dispatched).
"""
from __future__ import annotations

import numpy as np

from .types import MetricResult, TimeSeries


def curtailed_energy(ts: TimeSeries) -> MetricResult:
    avail_cols = [c for c in ts.columns if c.startswith("p_pv_avail_")]
    if not avail_cols:
        raise ValueError("no p_pv_avail_<id> columns found")

    t = ts["t"].to_numpy()
    total_kwh = 0.0
    per_asset: dict[str, float] = {}
    for ac in avail_cols:
        asset_id = ac.removeprefix("p_pv_avail_")
        pc = f"p_pv_{asset_id}"
        if pc not in ts.columns:
            continue
        gap = np.clip(ts[ac].to_numpy() - ts[pc].to_numpy(), 0.0, None)
        kwh = float(np.trapezoid(gap, t) / 3600.0)
        per_asset[asset_id] = kwh
        total_kwh += kwh

    return MetricResult(
        name="curtailed_energy",
        value=total_kwh,
        unit="kWh",
        detail={"per_asset_kwh": per_asset},
    )
