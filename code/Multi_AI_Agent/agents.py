"""Five agent functions for the upgraded DER-SecAgent loop.

Pipeline (paper §4.3):
    Telemetry Analyst (deterministic)
        → Hypothesis Agent (LLM, K=3 self-consistency)
        → Energy Impact Agent (grounded estimator)
        → Caution Agent (LLM safety review)
        → Coordinator Agent (deterministic policy + critic)
        → Report Agent (post-hoc, hash-chained)

Each agent consumes a state dict and writes back into it. Vendor APIs are
deliberately not used; the LoRA-served Qwen2.5-7B (or its heuristic fallback)
is the sole LLM backend.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..llm_serving.local_lora import get_default
from .coordinator import coordinator_agent as _coordinator_agent
from .energy_estimator import estimate as estimate_impacts
from .memory import IncidentMemory
from .self_consistency import run_self_consistent
from .states import AgentState, AttackClass, ImpactTier, OperationMode, PromptStrategy
from .telemetry_features import FeatureView, extract as extract_features

PROMPT_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Shared helpers

def _llm_json(prompt: str, role: str = "unspecified",
              sink: list[dict] | None = None) -> dict[str, Any]:
    """Single entry point to the LoRA-served LLM. Falls back to the
    discriminative heuristic when ML deps are not available (raises in
    ``DER_LLM_STRICT=1`` mode instead). When ``sink`` is provided, the
    per-call provenance trace (backend, adapter sha, prompt sha, raw output,
    latency, fallback reason) is appended to it under the given ``role``."""
    parsed, trace = get_default().generate_json_with_trace(prompt)
    if sink is not None:
        trace["role"] = role
        sink.append(trace)
    return parsed


def _read_prompt(name: str) -> tuple[str, str]:
    p = PROMPT_DIR / name
    s = p.read_text()
    return s, hashlib.sha256(s.encode()).hexdigest()[:12]


def _fill(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    return out


def _strategy_hint(strategy: PromptStrategy | str) -> str:
    val = strategy.value if hasattr(strategy, "value") else str(strategy)
    if val == "cot":
        return _read_prompt("cot.txt")[0]
    if val == "tot":
        return _read_prompt("tot.txt")[0]
    if val == "few":
        return "(few-shot exemplars; see Appendix A.5)"
    return ""


# ---------------------------------------------------------------------------
# Agent 1: Telemetry Analyst (deterministic; no LLM call)

def telemetry_analyst_agent(state: AgentState) -> AgentState:
    fv = extract_features(
        telemetry_window=state.get("telemetry_window", []) or [],
        event_window=state.get("event_window", []) or [],
        runtime_safe=bool(state.get("runtime_safe", False)),
    )
    state["feature_view"] = fv.to_dict()
    return state


# ---------------------------------------------------------------------------
# Agent 2: Hypothesis Agent (LLM, K=3 self-consistency over strategy variants)

def _compact_windows(state: AgentState) -> tuple[str, str, str]:
    """Render the FeatureView and recent windows in the *fine-tuning* format.

    Revision note (ijcip_revision_r1r2_20260805): the deployed
    ``hypothesis.txt`` embeds the raw FeatureView, five telemetry samples and
    twenty events as pretty JSON, which tokenises to ~2.4k tokens. The QLoRA
    adapters were fine-tuned on ~180-character prompts of the form
    ``Recent telemetry: ... / Recent events: ...``. Serving the long prompt is
    therefore both out-of-distribution for the adapter and, on a CPU substrate,
    minutes per call. This renderer reproduces the fine-tuning surface form
    from the same underlying state, so the in-loop configuration serves the
    model in-distribution. Nothing is hidden from the model that changes the
    decision inputs: the same dominant signal, asset, severity, per-asset
    power and event flags are present.
    """
    fv = state.get("feature_view", {}) or {}
    feat = (f"dominant_signal={fv.get('dominant_signal','none')} "
            f"dominant_asset={fv.get('dominant_asset')} "
            f"severity={float(fv.get('severity_score', 0.0)):.2f} "
            f"n_tampered={int(fv.get('n_tampered_events', 0))} "
            f"n_command={int(fv.get('n_command_events', 0))} "
            f"n_dup={int(fv.get('n_dup_events', 0))} "
            f"persistent_freeze={bool(fv.get('persistent_freeze', False))}")
    tw = state.get("telemetry_window", []) or []
    if tw:
        last = tw[-1]
        v = list((last.get("v_pu") or {}).values()) or [1.0]
        p = {k: round(float(x), 1) for k, x in (last.get("der_p_kw") or {}).items()}
        tel = (f"t={float(last.get('t', 0.0)):.0f}s freq={float(last.get('freq_hz', 60.0)):.2f}Hz "
               f"v_min_pu={min(v):.3f} v_max_pu={max(v):.3f} DER P kW={p}")
    else:
        tel = "(none)"
    evs = (state.get("event_window", []) or [])[-6:]
    if evs:
        lines = []
        for e in evs:
            pl = e.get("payload") or {}
            keep = {k: pl[k] for k in list(pl)[:3]}
            lines.append(f"- t={float(e.get('t', 0.0)):.1f} {e.get('source')}/{e.get('kind')}: "
                         f"{keep} tampered={bool(e.get('tampered', False))}")
        ev = "\n".join(lines)
    else:
        ev = "(none)"
    return feat, tel, ev


def hypothesis_agent(state: AgentState,
                     memory: IncidentMemory | None = None,
                     compact: bool = False) -> AgentState:
    template, _ = _read_prompt("hypothesis_compact.txt" if compact else "hypothesis.txt")
    fv = state.get("feature_view", {}) or {}

    memory_block = "(no prior incidents on file)"
    if memory is not None:
        kind = IncidentMemory.asset_kind_from_id(fv.get("dominant_asset"))
        prior = memory.lookup(fv.get("dominant_signal", "none"), kind)
        if prior:
            memory_block = json.dumps([
                {"attack_class": r.attack_class, "chosen_action": r.chosen_action,
                 "outcome_tier": r.outcome_tier, "confidence": round(r.confidence, 2)}
                for r in prior])

    def build_prompt(strat: str) -> str:
        if compact:
            feat, tel, ev = _compact_windows(state)
            return _fill(template, feature_summary=feat, telemetry_summary=tel,
                         event_summary=ev, memory=memory_block,
                         strategy_hint=_strategy_hint(strat))
        return _fill(
            template,
            feature_view=json.dumps(fv),
            telemetry_window=json.dumps(state.get("telemetry_window", [])[-5:],
                                        default=str),
            event_window=json.dumps(state.get("event_window", [])[-20:],
                                    default=str),
            strategy_hint=_strategy_hint(strat),
            memory=memory_block,
        )

    traces = state.setdefault("llm_traces", [])
    aggregate = run_self_consistent(
        build_prompt, lambda p: _llm_json(p, role="hypothesis", sink=traces),
        strategies=("zero",) if state.get("k_setting") == 1 else ("zero", "cot", "tot"))

    state["attack_class"] = AttackClass(aggregate.get("attack_class", "none"))
    state["affected_asset"] = (aggregate.get("affected_asset")
                                or fv.get("dominant_asset") or "")
    state["confidence"] = float(aggregate.get("confidence", 0.0))
    state["rationale"] = str(aggregate.get("rationale", ""))[:240]
    state["proposed_action"] = aggregate.get("proposed_action", "no_op")
    state["proposed_action_target"] = (aggregate.get("proposed_action_target")
                                        or state.get("affected_asset") or "")
    state["expected_impact_tier"] = ImpactTier(aggregate.get("expected_impact_tier",
                                                              "negligible"))
    state["hypothesis"] = aggregate
    state["decision_id"] = str(uuid.uuid4())
    return state


# ---------------------------------------------------------------------------
# Agent 3: Energy Impact Agent (grounded estimator).

def energy_impact_agent(state: AgentState) -> AgentState:
    fv_dict = state.get("feature_view", {}) or {}
    fv = FeatureView(
        t=fv_dict.get("t", 0.0),
        asset_zscores={k: float(v) for k, v in (fv_dict.get("asset_zscores") or {}).items()},
        asset_p_kw={k: float(v) for k, v in (fv_dict.get("asset_p_kw") or {}).items()},
        asset_p_avail_kw={k: float(v) for k, v in (fv_dict.get("asset_p_avail_kw") or {}).items()},
        voltage_excursion_frac=float(fv_dict.get("voltage_excursion_frac", 0.0)),
        voltage_min_pu=float(fv_dict.get("voltage_min_pu", 1.0)),
        voltage_max_pu=float(fv_dict.get("voltage_max_pu", 1.0)),
        freq_dev_hz=float(fv_dict.get("freq_dev_hz", 0.0)),
        n_command_events=int(fv_dict.get("n_command_events", 0)),
        n_tampered_events=int(fv_dict.get("n_tampered_events", 0)),
        n_dup_events=int(fv_dict.get("n_dup_events", 0)),
        persistent_freeze=bool(fv_dict.get("persistent_freeze", False)),
        dominant_asset=fv_dict.get("dominant_asset"),
        dominant_signal=fv_dict.get("dominant_signal", "none"),
        severity_score=float(fv_dict.get("severity_score", 0.0)),
    )
    target_asset = (state.get("affected_asset")
                    or state.get("proposed_action_target")
                    or fv.dominant_asset)
    estimates = estimate_impacts(fv, target_asset=target_asset)
    state["impact_estimates"] = [asdict(e) for e in estimates]
    return state


# ---------------------------------------------------------------------------
# Agent 4: Caution Agent (LLM safety review with deterministic floor)

def caution_eval_agent(state: AgentState) -> AgentState:
    template, _ = _read_prompt("caution.txt")
    proposed = {
        "attack_class": state.get("attack_class", AttackClass.NONE).value,
        "affected_asset": state.get("affected_asset"),
        "confidence": state.get("confidence", 0.0),
        "rationale": state.get("rationale", ""),
        "proposed_action": state.get("proposed_action", "no_op"),
        "proposed_action_target": state.get("proposed_action_target"),
        "expected_impact_tier": state.get("expected_impact_tier",
                                            ImpactTier.NEGLIGIBLE).value,
    }
    prompt = _fill(template, proposed_decision=json.dumps(proposed))
    out = _llm_json(prompt, role="caution", sink=state.setdefault("llm_traces", []))
    heuristic_flag = (
        state.get("proposed_action", "no_op") != "no_op"
        and state.get("expected_impact_tier", ImpactTier.NEGLIGIBLE)
              in {ImpactTier.HIGH, ImpactTier.SEVERE}
        and state.get("confidence", 0.0) < 0.7
    )
    state["caution"] = bool(out.get("caution", False)) or heuristic_flag
    state["caution_reason"] = str(out.get("caution_reason", "heuristic"))[:200]
    return state


# ---------------------------------------------------------------------------
# Agent 5: Coordinator Agent (deterministic critic — see coordinator.py)

def coordinator_agent(state: AgentState) -> AgentState:
    return _coordinator_agent(state)   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Agent 6: Report Agent (post-hoc, hash-chained)

def report_gen_agent(state: AgentState, prev_report_hash: str = "GENESIS",
                     actions_executed: list[dict] | None = None,
                     operator_response: dict | None = None) -> AgentState:
    template, _ = _read_prompt("report.txt")
    prompt = _fill(
        template,
        decision_json=json.dumps({k: (v.value if hasattr(v, "value") else v)
                                   for k, v in state.items()}, default=str),
        actions_executed_json=json.dumps(actions_executed or []),
        operator_response_json=json.dumps(operator_response or {"approved": True}),
        prev_report_hash=prev_report_hash,
    )
    out = _llm_json(prompt, role="report", sink=state.setdefault("llm_traces", []))
    state["report"] = json.dumps(out) if out else "{}"
    return state


# Backwards-compat alias (used by build_graph.py and some tests).
script_gen_agent = hypothesis_agent
