"""Attack injector interface."""
from __future__ import annotations

from typing import Any, Protocol

from ..types import EventLog, TelemetrySample


class AttackInjector(Protocol):
    name: str
    target: str           # asset id under attack
    start_s: float
    end_s: float

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample: ...

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]: ...

    def physical_mutate(self, t: float, feeder: Any) -> None:
        """Apply ground-truth physical consequence to the feeder.

        Default: no-op. FDI / replay attacks deliberately leave physical state
        untouched (their harm comes from the operator acting on bad data).
        Command-spoof and DoS implement real setpoint changes here.
        """
        return None
