"""Prior multi-agent baseline — two-agent (analyst + responder) loop.

Re-implements the 2-agent decomposition that has dominated open-source
LLM-SOC literature in 2024-2025: an Analyst Agent classifies the alert and
a Responder Agent picks a mitigation. There is no Caution Agent and no
HITL gate, so the responder always emits an action whenever the analyst
flags an attack at or above ``min_confidence_to_act``. This matches the
"prior MAS" referent we compare against in §5.6 of the paper.

Behavioural distinction from ``single_llm``:
  - Single-LLM aggressively isolates inverters on FDI (full curtailment).
  - Prior-MAS responder prefers ``freeze_setpoint`` on FDI (less
    curtailment, but still acts every time the analyst flags) — a slightly
    more sophisticated default, consistent with how 2-agent SOC papers
    typically constrain the responder.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ...llm_serving.local_lora import LocalLoRA, get_default
from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample
from ..single_llm.runner import _format_events, _summarize_telemetry


ANALYST_PROMPT = """You are the Analyst Agent. Classify the recent telemetry
and event window. Respond with JSON only:
{
  "attack_class": "none|fdi|replay|command_spoof|dos|firmware",
  "affected_asset": "<asset id or null>",
  "confidence": 0.0-1.0,
  "rationale": "<short reason>"
}

Recent telemetry:
{telemetry_summary}

Recent events:
{event_log}
"""

RESPONDER_PROMPT = """You are the Responder Agent. Given the analyst's
classification, pick a mitigation primitive. Respond with JSON only:
{
  "recommended_action": "no_op|isolate_inverter|throttle_ramp|freeze_setpoint|request_ied_revalidation"
}

Analyst classification: {analyst_json}
"""


@dataclass
class PriorMAS:
    name: str = "prior_mas"
    min_confidence_to_act: float = 0.45
    history_len: int = 10
    _telemetry_buf: deque = field(default_factory=lambda: deque(maxlen=10))
    _llm_backend: LocalLoRA | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._llm_backend = get_default()

    def step(
        self,
        t: float,
        telemetry: TelemetrySample,
        events: list[EventLog],
    ) -> tuple[Detection, list[Action]]:
        self._telemetry_buf.append(telemetry)
        analyst_prompt = (
            ANALYST_PROMPT
            .replace("{telemetry_summary}", _summarize_telemetry(list(self._telemetry_buf)))
            .replace("{event_log}", _format_events(events))
        )
        analyst = self._llm_backend.generate_json(analyst_prompt)
        det = Detection(
            asset=analyst.get("affected_asset"),
            attack_class=analyst.get("attack_class", "none"),
            confidence=float(analyst.get("confidence", 0.0)),
            rationale=str(analyst.get("rationale", ""))[:240],
        )
        if det.attack_class == "none" or det.confidence < self.min_confidence_to_act:
            return det, []

        # Responder picks an action. Prior-MAS responder is conservative on FDI.
        responder_prompt = RESPONDER_PROMPT.replace("{analyst_json}", str(analyst))
        responder = self._llm_backend.generate_json(responder_prompt)
        action_name = responder.get("recommended_action", "freeze_setpoint")
        if det.attack_class == "fdi":
            action_name = "freeze_setpoint"   # 2-agent default
        if action_name not in MITIGATIONS or action_name == "no_op":
            return det, []
        return det, [Action(name=action_name, target=det.asset or "")]
