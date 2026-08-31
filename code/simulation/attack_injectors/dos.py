"""Denial-of-Service injector.

Models loss of telemetry + a dropout in DERMS dispatch. Telemetry sees the
last-good values frozen; physically, the targeted asset trips offline (kW = 0)
because its dispatch controller cannot reach the EMS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import EventLog, TelemetrySample


@dataclass
class DoSInjector:
    target: str
    start_s: float
    end_s: float
    name: str = "dos"
    _last_good: TelemetrySample | None = field(default=None, repr=False)

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample:
        if not self.active(t):
            self._last_good = sample
            return sample
        return self._last_good or sample

    def physical_mutate(self, t: float, feeder: Any) -> None:
        if not self.active(t):
            return
        if hasattr(feeder, "set_generator_kw"):
            feeder.set_generator_kw(self.target, 0.0)

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]:
        if not self.active(t):
            return events
        return [e for e in events if e.payload.get("asset") != self.target]
