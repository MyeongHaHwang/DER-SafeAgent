"""Shared types for energy-impact metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    unit: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {"metric": self.name, "value": self.value, "unit": self.unit}
        row.update({f"detail.{k}": v for k, v in self.detail.items()})
        return row


# Type alias — a TimeSeries DataFrame must contain at minimum:
#   t              [s]  — sample time (monotonic)
#   load_demand    [kW] — total demanded load at the feeder head (or per-bus if available)
#   load_served    [kW] — actually served load
#   v_pu_<bus>     [pu] — per-bus voltage magnitude (one column per monitored bus)
#   freq           [Hz] — system frequency (single column)
#   p_pv_<id>      [kW] — PV active power dispatched (per asset)
#   p_pv_avail_<id>[kW] — PV active power available (max, no curtailment)
#   p_bess_<id>    [kW] — BESS active power (negative = charging)
# Additional columns are ignored.
TimeSeries = pd.DataFrame
