"""Single-LLM baseline.

Wraps a locked prompt and the LoRA-finetuned model served by
`code/llm_serving/local_lora.py`. No tool use, no agent loop. Same decision
surface as the proposed system so the harness can run both with identical
scenarios for an apples-to-apples comparison.

Vendor APIs (Anthropic, OpenAI, etc.) are deliberately not used; this baseline
runs the *base* LoRA-tuned model without the multi-agent loop, which is the
ablation contrast described in §5.6 of the paper.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ...llm_serving.local_lora import LocalLoRA, get_default
from ...simulation.mitigations import REGISTRY as MITIGATIONS
from ...simulation.types import Action, Detection, EventLog, TelemetrySample

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def _summarize_telemetry(samples: list[TelemetrySample]) -> str:
    if not samples:
        return "(none)"
    last = samples[-1]
    return (
        f"t={last.t:.1f}s, freq={last.freq_hz:.3f} Hz, "
        f"v_min_pu={min(last.bus_voltages_pu.values()):.3f}, "
        f"v_max_pu={max(last.bus_voltages_pu.values()):.3f}, "
        f"DER P kW={ {k: round(v, 1) for k, v in last.der_p_kw.items()} }"
    )


def _format_events(events: list[EventLog]) -> str:
    def fmt(e: EventLog) -> str:
        # Render every event as a single JSON object so the downstream tokenizer
        # (and our heuristic fallback) sees the canonical ``"tampered": true``
        # marker rather than a freeform " [tampered]" suffix it has to parse.
        rec = {"t": e.t, "source": e.source, "kind": e.kind,
               "payload": e.payload, "tampered": e.tampered}
        return f"- {json.dumps(rec)}"
    return "\n".join(fmt(e) for e in events[-20:]) or "(none)"


@dataclass
class SingleLLM:
    """Single-prompt baseline: no agent loop, no caution gate.

    Acts whenever the model says attack_class != "none" — this is the
    "trust the LLM" baseline the paper compares against. Aggressive policy:
    on FDI, escalates the recommendation to ``isolate_inverter`` (worst-case
    over-mitigation) rather than ``freeze_setpoint``; on command_spoof,
    keeps ``freeze_setpoint``. The escalation surfaces the curtailment cost
    of acting without a safety review, which is the single-LLM/agent-loop
    contrast in §5.6.
    """
    name: str = "single_llm"
    history_len: int = 10
    min_confidence_to_act: float = 0.5
    # Incident-level invocation gate for real-model-in-the-loop runs (B1):
    # ``None`` keeps the published behaviour (one LLM call per tick). A number
    # N > 0 calls the model only on ticks that carry events, at most once per
    # N seconds, reusing the previous model output in between. Provenance for
    # every real call is recorded via ``_last_decision["llm_traces"]``.
    invoke_interval_s: float | None = None
    _telemetry_buf: deque = field(default_factory=lambda: deque(maxlen=10))
    _llm_backend: LocalLoRA | None = field(default=None, repr=False)
    _last_out: dict | None = field(default=None, repr=False)
    _last_invoke_t: float | None = field(default=None, repr=False)
    _last_decision: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._prompt_template = PROMPT_PATH.read_text()
        self._prompt_hash = hashlib.sha256(self._prompt_template.encode()).hexdigest()[:12]
        self._llm_backend = get_default()

    def step(
        self,
        t: float,
        telemetry: TelemetrySample,
        events: list[EventLog],
    ) -> tuple[Detection, list[Action]]:
        self._telemetry_buf.append(telemetry)
        gated = self.invoke_interval_s is not None
        if gated and not events:
            # Quiet tick under the incident gate: no evidence to assess.
            self._last_decision = None
            return Detection(asset=None, attack_class="none", confidence=0.0), []
        prompt = (
            self._prompt_template
            .replace("{telemetry_summary}", _summarize_telemetry(list(self._telemetry_buf)))
            .replace("{event_log}", _format_events(events))
        )
        traces: list[dict] = []
        cached = False
        if (gated and self._last_out is not None
                and self._last_invoke_t is not None
                and (t - self._last_invoke_t) < self.invoke_interval_s):
            out = self._last_out
            cached = True
        else:
            out, trace = self._llm_backend.generate_json_with_trace(prompt)
            trace["role"] = "single_llm"
            traces.append(trace)
            if gated:
                self._last_out = out
                self._last_invoke_t = t
        det = Detection(
            asset=out.get("affected_asset"),
            attack_class=out.get("attack_class", "none"),
            confidence=float(out.get("confidence", 0.0)),
            rationale=str(out.get("rationale", ""))[:240],
        )
        # Aggressive policy: FDI escalates to isolate_inverter (cuts the asset
        # entirely → high curtailment cost). Other classes keep the proposed
        # primitive but skip any caution review.
        proposed = out.get("recommended_action", "no_op")
        action_name = proposed
        if det.attack_class == "fdi":
            action_name = "isolate_inverter"
        act = (det.attack_class != "none"
               and det.confidence >= self.min_confidence_to_act
               and action_name in MITIGATIONS and action_name != "no_op")
        self._last_decision = {
            "t": t,
            "proposed_action": proposed,
            "proposed_action_target": out.get("affected_asset"),
            "confidence": det.confidence,
            "final_action": action_name if act else "no_op",
            "final_action_target": out.get("affected_asset") if act else None,
            "coordinator_decision": "auto",
            "hypothesis_cached": cached,
            "llm_traces": traces,
        }
        if not act:
            return det, []
        return det, [Action(name=action_name, target=out.get("affected_asset") or "")]
