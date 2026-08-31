"""DeterministicEnergyPolicy --- no-LLM baseline.

Uses the same Telemetry-Analyst feature view and class-aware mitigation
table as DER-SecAgent, but skips every LLM call. The intent is to isolate
the marginal value of the LLM agents (Hypothesis, Energy Impact, Caution)
on top of the deterministic feature pipeline.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ...Multi_AI_Agent.telemetry_features import extract as extract_features
from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample


_CLASS_FROM_SIGNAL: dict[str, str] = {
    "command":            "command_spoof",
    "tampered":           "fdi",   # legacy oracle signal (non-runtime-safe)
    "integrity_residual": "fdi",   # v3 observable replacement for `tampered`
    "replay":             "replay",
    "freeze":             "dos",
    "telemetry_dev":      "fdi",
    "none":               "none",
}

_FORCED_ACTION_BY_CLASS: dict[str, str] = {
    "command_spoof": "freeze_setpoint",
    "fdi":           "freeze_setpoint",
    "replay":        "request_ied_revalidation",
    "dos":           "request_ied_revalidation",
    "firmware":      "request_ied_revalidation",
    "none":          "no_op",
}


@dataclass
class DeterministicEnergyPolicy:
    name: str = "deterministic_energy_policy"
    telemetry_buffer_len: int = 60
    event_buffer_len: int = 60
    severity_threshold: float = 0.30
    runtime_safe: bool = False   # v3: ignore injector `tampered` oracle
    _telemetry: deque = field(default_factory=lambda: deque(maxlen=60))
    _events: deque = field(default_factory=lambda: deque(maxlen=60))
    _last_decision: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._telemetry = deque(maxlen=self.telemetry_buffer_len)
        self._events = deque(maxlen=self.event_buffer_len)

    def _tel_dict(self, s: TelemetrySample) -> dict:
        return {"t": s.t, "freq_hz": s.freq_hz,
                 "v_pu": s.bus_voltages_pu, "der_p_kw": s.der_p_kw,
                 "der_p_avail": s.der_p_avail_kw,
                 "load_demand_kw": s.load_demand_kw,
                 "load_served_kw": s.load_served_kw}

    def _ev_dict(self, e: EventLog) -> dict:
        return {"t": e.t, "source": e.source, "kind": e.kind,
                 "payload": e.payload, "tampered": e.tampered, "seq": getattr(e, "seq", None)}

    def step(self, t: float, telemetry: TelemetrySample,
             events: list[EventLog]) -> tuple[Detection, list[Action]]:
        self._telemetry.append(telemetry)
        for ev in events:
            self._events.append(ev)
        fv = extract_features(
            telemetry_window=[self._tel_dict(s) for s in self._telemetry],
            event_window=[self._ev_dict(e) for e in self._events],
            runtime_safe=self.runtime_safe,
        )
        if fv.severity_score < self.severity_threshold:
            self._last_decision = {"feature_view": fv.to_dict(),
                                     "attack_class": "none",
                                     "final_action": "no_op"}
            return Detection(asset=None, attack_class="none",
                              confidence=0.0), []
        cls = _CLASS_FROM_SIGNAL.get(fv.dominant_signal, "none")
        action_name = _FORCED_ACTION_BY_CLASS.get(cls, "no_op")
        target = fv.dominant_asset or ""
        det = Detection(asset=target or None, attack_class=cls,
                          confidence=fv.severity_score,
                          rationale=f"deterministic|{fv.dominant_signal}")
        if action_name == "no_op" or action_name not in MITIGATIONS:
            self._last_decision = {"attack_class": cls, "final_action": "no_op"}
            return det, []
        self._last_decision = {"feature_view": fv.to_dict(),
                                 "attack_class": cls,
                                 "final_action": action_name}
        return det, [Action(name=action_name, target=target,
                              params={"detector": "deterministic_energy_policy"})]
