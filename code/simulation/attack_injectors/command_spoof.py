"""Command-spoof: injects forged DERMS dispatch commands targeting an asset."""
from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from ..types import EventLog, TelemetrySample


@dataclass
class CommandSpoofInjector:
    target: str
    start_s: float
    end_s: float
    forged_setpoint_kw: float
    period_s: float = 5.0
    name: str = "command_spoof"

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample:
        return sample

    def physical_mutate(self, t: float, feeder: Any) -> None:
        """Force the targeted Generator's actual kW to the forged setpoint."""
        if not self.active(t):
            return
        if hasattr(feeder, "set_generator_kw"):
            feeder.set_generator_kw(self.target, self.forged_setpoint_kw)

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]:
        if not self.active(t):
            return events
        if int(t) % int(max(1, self.period_s)) != 0:
            return events
        forged = EventLog(
            t=t,
            source="dnp3",
            kind="command",
            # auth_valid=False: a forged command carries no valid MAC/signature.
            # A gate with authenticated command evidence can reject it; on an
            # unauthenticated bus it is observationally identical to a
            # legitimate command (see benign_command injector).
            payload={"asset": self.target, "type": "setpoint",
                     "p_kw": self.forged_setpoint_kw, "from": "DERMS",
                     "auth_valid": False},
            tampered=True,
        )
        return events + [forged]
