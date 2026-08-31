"""Inverter ramp-rate violations per IEEE 1547 normal-ramp-rate.

A violation is counted when the magnitude of dP/dt for any monitored DER
exceeds the configured limit. Default limit is a project-set proxy and should
be overridden per asset class in scenario configs.
"""
from __future__ import annotations

import numpy as np

from .types import MetricResult, TimeSeries

DEFAULT_LIMIT_KW_PER_S = 1.667  # ~ 100 kW/min as conservative proxy


def ramp_violations(
    ts: TimeSeries,
    limit_kw_per_s: float = DEFAULT_LIMIT_KW_PER_S,
) -> MetricResult:
    der_cols = [c for c in ts.columns if c.startswith(("p_pv_", "p_bess_")) and "avail" not in c]
    if not der_cols:
        raise ValueError("no p_pv_<id> or p_bess_<id> columns found")

    t = ts["t"].to_numpy()
    if len(t) < 2:
        return MetricResult(name="ramp_violations", value=0.0, unit="count",
                            detail={"limit_kw_per_s": limit_kw_per_s, "per_asset": {}})

    dt = np.diff(t)
    per_asset: dict[str, int] = {}
    total = 0
    for c in der_cols:
        p = ts[c].to_numpy()
        rate = np.abs(np.diff(p) / dt)
        n = int((rate > limit_kw_per_s).sum())
        per_asset[c] = n
        total += n

    return MetricResult(
        name="ramp_violations",
        value=float(total),
        unit="count",
        detail={"limit_kw_per_s": limit_kw_per_s, "per_asset": per_asset},
    )
