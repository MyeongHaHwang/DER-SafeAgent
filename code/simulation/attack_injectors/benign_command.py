"""Benign legitimate DERMS command bursts (NOT an attack).

Models routine authenticated dispatch commands (setpoint adjustments) that a
real DERMS issues during normal operation. These carry a valid MAC
(auth_valid=True) and DO NOT change the physical state adversarially --- they
apply a small, legitimate setpoint nudge within nominal bounds. Their purpose
is to make command-based detection non-trivial: a gate that treats ANY command
packet as an attack will false-open on these, and only a gate that checks
command authentication can tell them apart from a spoof
(attack_injectors/command_spoof.py). Ground truth for scenarios using this
injector is `none`.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import EventLog, TelemetrySample


@dataclass
class BenignCommandInjector:
    target: str
    start_s: float
    end_s: float
    setpoint_kw: float = 140.0   # within nominal (0.7 * rating for a 200 kW PV)
    period_s: float = 5.0
    name: str = "benign_command"

    def active(self, t: float) -> bool:
        return False   # never an attack window for ground truth

    def _bursting(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample:
        return sample

    def physical_mutate(self, t: float, feeder) -> None:
        # a legitimate in-bounds setpoint; small, non-adversarial
        if self._bursting(t) and hasattr(feeder, "set_generator_kw"):
            feeder.set_generator_kw(self.target, self.setpoint_kw)

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]:
        if not self._bursting(t) or int(t) % int(max(1, self.period_s)) != 0:
            return events
        legit = EventLog(
            t=t, source="dnp3", kind="command",
            payload={"asset": self.target, "type": "setpoint",
                     "p_kw": self.setpoint_kw, "from": "DERMS",
                     "auth_valid": True},
            tampered=False,
        )
        return events + [legit]
