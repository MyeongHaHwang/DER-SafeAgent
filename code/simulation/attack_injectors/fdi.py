"""False Data Injection on inverter telemetry.

Adds a configurable bias to the reported PV active power for one asset, so the
DERMS believes the asset is generating more (or less) than it actually is. The
ground-truth power in OpenDSS is unaffected; only what the detector sees is.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..types import EventLog, TelemetrySample


@dataclass
class FDIInjector:
    target: str
    start_s: float
    end_s: float
    bias_kw: float
    name: str = "fdi"

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s

    def mutate_telemetry(self, t: float, sample: TelemetrySample) -> TelemetrySample:
        if not self.active(t) or self.target not in sample.der_p_kw:
            return sample
        new_p = dict(sample.der_p_kw)
        new_p[self.target] = sample.der_p_kw[self.target] + self.bias_kw
        return replace(sample, der_p_kw=new_p)

    def physical_mutate(self, t: float, feeder) -> None:
        """FDI is a telemetry-layer attack only; no physical state change."""
        return None

    def mutate_events(self, t: float, events: list[EventLog]) -> list[EventLog]:
        if not self.active(t):
            return events
        out: list[EventLog] = []
        for ev in events:
            if ev.kind == "telemetry" and ev.payload.get("asset") == self.target:
                tampered = dict(ev.payload)
                tampered["p_kw"] = tampered.get("p_kw", 0.0) + self.bias_kw
                out.append(EventLog(t=ev.t, source=ev.source, kind=ev.kind,
                                    payload=tampered, tampered=True))
            else:
                out.append(ev)
        return out
