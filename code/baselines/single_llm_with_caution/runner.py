"""SingleLLMWithCaution --- single-LLM detector + Caution Agent veto.

Adds the four-rule Caution Agent on top of the existing single-LLM
baseline. When the Caution Agent flags an action *and* its expected impact
tier is high or severe with confidence below the floor, the action is
withheld. This isolates the marginal value of the safety review without
the rest of the multi-agent decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...Multi_AI_Agent.agents import caution_eval_agent
from ...Multi_AI_Agent.states import AttackClass, ImpactTier
from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample
from ..single_llm.runner import SingleLLM


_HIGH = {ImpactTier.HIGH, ImpactTier.SEVERE}


@dataclass
class SingleLLMWithCaution:
    name: str = "single_llm_with_caution"
    high_impact_confidence_floor: float = 0.7
    _inner: SingleLLM = field(default_factory=SingleLLM)
    _last_decision: dict | None = field(default=None, repr=False)

    def step(self, t: float, telemetry: TelemetrySample,
             events: list[EventLog]) -> tuple[Detection, list[Action]]:
        det, actions = self._inner.step(t, telemetry, events)
        if det.attack_class == "none" or not actions:
            return det, []

        tier_str = "medium"
        # Build a state-like dict for the Caution Agent.
        state = {
            "attack_class": AttackClass(det.attack_class),
            "affected_asset": det.asset or actions[0].target,
            "confidence": float(det.confidence),
            "rationale": det.rationale,
            "proposed_action": actions[0].name,
            "proposed_action_target": actions[0].target,
            "expected_impact_tier": ImpactTier.HIGH if actions[0].name == "isolate_inverter" else ImpactTier.MEDIUM,
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
        return det, actions
