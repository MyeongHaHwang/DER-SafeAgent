"""Benign telemetry echo — NOT an attack.

Models a mundane OT data-path quirk: a SCADA historian / gateway retransmits
telemetry events, producing duplicated payloads with no tampering and no
physical effect. Used by the P0-B Incident Evidence Gate evaluation to place
protocol-level (hard-evidence) ambiguity inside BENIGN data, so the gate is
not trivially separable by construction. Ground truth for scenarios using
this injector is ``none``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import EventLog, TelemetrySample


@dataclass
class BenignEchoInjector:
    target: str
    start_s: float
    end_s: float
    period_s: float = 3.0     # retransmit every N seconds
    name: str = "benign_echo"

    def active(self, t: float) -> bool:
        # Benign: never counts as an active attack for ground truth.
        return False

    def _echoing(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample:
        return sample

    def physical_mutate(self, t: float, feeder) -> None:
        return None

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]:
        if not self._echoing(t) or int(t) % int(max(1, self.period_s)) != 0:
            return events
        echoes = [EventLog(t=e.t, source=e.source, kind=e.kind,
                           payload=dict(e.payload), tampered=False)
                  for e in events
                  if e.kind == "telemetry"
                  and e.payload.get("asset") == self.target]
        return events + echoes
