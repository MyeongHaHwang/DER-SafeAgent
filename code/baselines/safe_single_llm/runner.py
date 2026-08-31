"""SafeSingleLLM --- Single-LLM detector with the same safety wrappers as
DER-SecAgent: forbidden-action filter and class-aware avoidance set.

Used in the IJCIP revision to test whether the safety layer alone (without
the multi-agent decomposition) closes the gap to DER-SecAgent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample
from ..single_llm.runner import SingleLLM


_AVOID_BY_CLASS: dict[str, set[str]] = {
    "fdi":           {"isolate_inverter"},
    "replay":        {"isolate_inverter"},
    "command_spoof": set(),
    "dos":           set(),
    "firmware":      set(),
    "none":          {"isolate_inverter", "freeze_setpoint"},
}

_FORCED_BY_CLASS: dict[str, str] = {
    "fdi":           "freeze_setpoint",
    "command_spoof": "freeze_setpoint",
    "replay":        "request_ied_revalidation",
    "dos":           "request_ied_revalidation",
    "firmware":      "request_ied_revalidation",
}


@dataclass
class SafeSingleLLM:
    """Single-LLM detector with the DER-SecAgent safety wrappers."""
    name: str = "safe_single_llm"
    _inner: SingleLLM = field(default_factory=SingleLLM)

    def __post_init__(self) -> None:
        # Reuse the upstream's prompt/buffers but override the action emission.
        pass

    @property
    def _last_decision(self) -> dict | None:                # for HITL routing
        return getattr(self._inner, "_last_decision", None)

    def step(self, t: float, telemetry: TelemetrySample,
             events: list[EventLog]) -> tuple[Detection, list[Action]]:
        det, actions = self._inner.step(t, telemetry, events)
        if det.attack_class == "none":
            return det, []
        # Apply class-aware avoidance + forced override.
        proposed = actions[0].name if actions else "no_op"
        avoid = _AVOID_BY_CLASS.get(det.attack_class, set())
        if proposed in avoid or proposed not in MITIGATIONS:
            forced = _FORCED_BY_CLASS.get(det.attack_class, "freeze_setpoint")
            if forced not in MITIGATIONS:
                return det, []
            proposed = forced
        # Emit a single safe action.
        target = actions[0].target if actions else (det.asset or "")
        return det, [Action(name=proposed, target=target,
                              params={"safety_layer": "class_aware_avoidance"})]
