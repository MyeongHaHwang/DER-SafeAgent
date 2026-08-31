"""PriorMASWithSafety --- 2-agent baseline + Caution + forbidden-action filter.

Wraps the existing PriorMAS to (a) drop forbidden actions, (b) enforce a
class-aware avoidance set (FDI / replay never receive isolate_inverter),
and (c) forward suspicious actions through the Caution Agent. Used to
isolate the marginal value of the safety review on top of a 2-agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...Multi_AI_Agent.agents import caution_eval_agent
from ...Multi_AI_Agent.states import AttackClass, ImpactTier
from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample
from ..prior_mas.adapter import PriorMAS


_AVOID_BY_CLASS: dict[str, set[str]] = {
    "fdi":     {"isolate_inverter"},
    "replay":  {"isolate_inverter"},
    "none":    {"isolate_inverter", "freeze_setpoint",
                "throttle_ramp", "request_ied_revalidation"},
}
_HIGH = {ImpactTier.HIGH, ImpactTier.SEVERE}


@dataclass
class PriorMASWithSafety:
    name: str = "prior_mas_with_safety"
    high_impact_confidence_floor: float = 0.7
    _inner: PriorMAS = field(default_factory=PriorMAS)
    _last_decision: dict | None = field(default=None, repr=False)

    def step(self, t: float, telemetry: TelemetrySample,
             events: list[EventLog]) -> tuple[Detection, list[Action]]:
        det, actions = self._inner.step(t, telemetry, events)
        if det.attack_class == "none" or not actions:
            return det, []

        avoid = _AVOID_BY_CLASS.get(det.attack_class, set())
        action = actions[0]
        if action.name in avoid:
            # downgrade to freeze_setpoint as the safe default.
            action = Action(name="freeze_setpoint", target=action.target,
                              params={"safety_layer": "class_aware_avoidance"})
        if action.name not in MITIGATIONS:
            return det, []

        # Caution Agent gate.
        state = {
            "attack_class": AttackClass(det.attack_class),
            "affected_asset": det.asset or action.target,
            "confidence": float(det.confidence),
            "rationale": det.rationale,
            "proposed_action": action.name,
            "proposed_action_target": action.target,
            "expected_impact_tier": ImpactTier.HIGH if action.name == "isolate_inverter" else ImpactTier.MEDIUM,
        }
        try:
            state = caution_eval_agent(state)
        except Exception:
            state["caution"] = False
        cautioned = bool(state.get("caution", False))
        impact = state.get("expected_impact_tier", ImpactTier.MEDIUM)
        if cautioned and impact in _HIGH and det.confidence < self.high_impact_confidence_floor:
            self._last_decision = {**state, "caution_route": "withhold"}
            return det, []
        self._last_decision = state
        return det, [action]
