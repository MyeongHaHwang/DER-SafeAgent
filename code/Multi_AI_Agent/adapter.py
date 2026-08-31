"""Detector Protocol adapter — wires the 5-agent loop into the harness.

Differences from the v1 adapter (paper §4):
  1. Trigger logic considers tampered-event prevalence, voltage excursion and
     persistent freeze, not only z-score and alarm/command events; the v1
     trigger missed sustained attacks once the rolling mean caught up.
  2. Pipeline runs Telemetry Analyst → Hypothesis (K=3 self-consistency)
     → Energy Impact → Caution → Coordinator.
  3. Action emission is driven by the Coordinator's ``final_action`` rather
     than the Hypothesis Agent's proposal — this is the energy-impact-aware
     policy hook required by §4.4.
  4. HITL routing reads ``coordinator_decision == "hitl_required"`` and
     forwards the held-back primitive to the harness operator stub.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..simulation.mitigations import REGISTRY as MITIGATIONS
from ..simulation.types import Action, Detection, EventLog, TelemetrySample
from .agents import (
    caution_eval_agent,
    coordinator_agent,
    energy_impact_agent,
    hypothesis_agent,
    telemetry_analyst_agent,
)
from .memory import IncidentMemory, IncidentRecord
from .states import AgentState, AttackClass, ImpactTier, OperationMode, PromptStrategy


def _telemetry_to_dict(s: TelemetrySample) -> dict[str, Any]:
    return {
        "t": s.t,
        "freq_hz": s.freq_hz,
        "v_pu": s.bus_voltages_pu,
        "der_p_kw": s.der_p_kw,
        "der_p_avail": s.der_p_avail_kw,
        "load_demand_kw": s.load_demand_kw,
        "load_served_kw": s.load_served_kw,
    }


def _event_to_dict(e: EventLog) -> dict[str, Any]:
    return {"t": e.t, "source": e.source, "kind": e.kind, "payload": e.payload,
            "tampered": e.tampered, "seq": getattr(e, "seq", None)}


ABLATION_FLAGS: tuple[str, ...] = (
    "full",
    "no_caution_agent",
    "no_energy_impact_agent",
    "no_class_aware_avoidance",
    "no_self_consistency",
    "no_incident_memory",
    "no_hitl_fallback",
    "deterministic_only",
)


@dataclass
class DERSecAgentDetector:
    """Multi-agent detector for DER cybersecurity decisions.

    The ``ablation`` field selectively disables individual agents or policy
    hooks. ``full`` is the production configuration; the other flags are
    used in the IJCIP ablation table (Sec.~5.X).
    """
    name: str = "der_secagent"
    prompt_strategy: PromptStrategy = PromptStrategy.COT
    telemetry_buffer_len: int = 60
    event_buffer_len: int = 60
    trigger_z_threshold: float = 3.0
    min_confidence_to_act: float = 0.4
    high_impact_confidence_floor: float = 0.7
    memory_path: str | None = None
    ablation: str = "full"
    # Incident-level LLM invocation policy. ``None`` (default, and the
    # configuration used for all published heuristic-backbone results) invokes
    # the Hypothesis/Caution stages on every triggered tick. A number N > 0
    # invokes them at the first triggered tick of an incident and then reuses
    # the cached hypothesis until N seconds elapse or the FeatureView's
    # dominant signal changes; the deterministic layers (Telemetry Analyst,
    # Energy Impact, Coordinator, avoidance, HITL gate) still run every tick.
    # This is the trigger-only serving posture described in the paper and is
    # what makes real-model-in-the-loop evaluation tractable.
    llm_invoke_interval_s: float | None = None
    # K setting for the Hypothesis Agent: 3 (published default, three prompting
    # strategies + majority vote) or 1 (single greedy call).
    k_setting: int = 3
    # Serve the Hypothesis Agent the compact, fine-tuning-distribution prompt
    # (see agents._compact_windows). Required for real-LoRA configurations;
    # False reproduces the published long-prompt behaviour exactly.
    compact_prompt: bool = False
    # Runtime-assurance policy (revision ijcip_final_revision, E1):
    #   "legacy_class_override" - published behaviour: the predicted class
    #       alone determines the action; the model's proposal is discarded.
    #   "safety_projection"     - the shield removes unsafe candidates and the
    #       model chooses among those that remain, so it has a causal path to
    #       the executed action while staying inside the bounded set.
    policy_mode: str = "legacy_class_override"
    # Deterministic-fallback ranker inside the safety projection (P0-A,
    # tag ijcip_final_safeagent_20260810). "class_table" reproduces the legacy
    # class-preference order; "eh" ranks the safe candidate set with the
    # holdout-validated horizon-aware estimator (EH). Only consulted when the
    # projection cannot honour the model's proposal.
    fallback_estimator: str = "class_table"
    # v3 Phase 1: ignore the injector `tampered` oracle in the FeatureView and
    # use the observable integrity residual instead (see telemetry_features).
    runtime_safe: bool = False

    _telemetry: deque = field(default_factory=lambda: deque(maxlen=60))
    _events: deque = field(default_factory=lambda: deque(maxlen=60))
    _last_decision: AgentState | None = field(default=None, repr=False)
    _memory: IncidentMemory | None = field(default=None, repr=False)
    _consecutive_zero: dict[str, int] = field(default_factory=dict, repr=False)
    _cached_llm_stage: dict | None = field(default=None, repr=False)
    _cached_signal: str | None = field(default=None, repr=False)
    _last_llm_invoke_t: float | None = field(default=None, repr=False)
    _incident_start_t: float | None = field(default=None, repr=False)
    _last_trigger_t: float | None = field(default=None, repr=False)
    _eh_calib: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._telemetry = deque(maxlen=self.telemetry_buffer_len)
        self._events = deque(maxlen=self.event_buffer_len)
        self._memory = IncidentMemory(
            store_path=Path(self.memory_path) if self.memory_path else None)

    def _should_invoke(self, telemetry: TelemetrySample, events: list[EventLog]) -> bool:
        # 1. Tampered events in this batch (fires every step of an active
        #    FDI / replay attack — fixes the v1 trigger that lost them once
        #    the rolling mean caught up to the bias).
        if any(getattr(ev, "tampered", False) for ev in events):
            return True
        # 2. Alarm or command events.
        if any(ev.kind in {"alarm", "command"} for ev in events):
            return True
        # 3. Voltage band excursion in the latest sample.
        for v in telemetry.bus_voltages_pu.values():
            if v < 0.95 or v > 1.05:
                return True
        # 4. Rolling-window z-score on any asset's reported P.
        if len(self._telemetry) >= 5 and telemetry.der_p_kw:
            for asset, p_now in telemetry.der_p_kw.items():
                history = [s.der_p_kw.get(asset, p_now) for s in self._telemetry]
                mean = sum(history) / len(history)
                var = sum((x - mean) ** 2 for x in history) / max(len(history), 1)
                std = (var ** 0.5) or 1e-6
                if abs(p_now - mean) / std > self.trigger_z_threshold:
                    return True
        # 5. SCADA staleness: reported timestamps stop advancing (DoS/replay
        #    freeze the whole sample at capture time). P0-B addition — the
        #    published triggers 1-4 and 6 are unchanged.
        if len(self._telemetry) >= 4:
            tail = [s.t for s in list(self._telemetry)[-4:]]
            if all(abs(tail[i + 1] - tail[i]) < 1e-9 for i in range(3)):
                return True
        # 6. Persistent freeze: same asset at zero kW for many ticks.
        for asset, p_now in telemetry.der_p_kw.items():
            if abs(p_now) < 1e-3:
                self._consecutive_zero[asset] = self._consecutive_zero.get(asset, 0) + 1
            else:
                self._consecutive_zero[asset] = 0
            if self._consecutive_zero[asset] >= 10:
                return True
        return False

    def _eh_fallback_choice(self, state: dict, t: float) -> str | None:
        """Rank the candidate registry with the holdout-validated EH estimator
        (P0-A). Called only when the projection falls back deterministically.
        """
        try:
            from .horizon_estimator import (HorizonCalibration, class_family,
                                            estimate_eh, select)
            if self._eh_calib is None:
                from pathlib import Path as _P
                p = _P("code/configs/ijcip_final_safeagent_20260810/"
                       "eh_duration_prior.json")
                if not p.exists():
                    return None
                self._eh_calib = HorizonCalibration.load(p)
            fv = state.get("feature_view", {}) or {}
            target = (state.get("affected_asset")
                      or fv.get("dominant_asset") or "")
            tw = state.get("telemetry_window") or []
            load_kw = float(tw[-1].get("load_demand_kw", 0.0)) if tw else None
            elapsed = max(0.0, t - (self._incident_start_t
                                    if self._incident_start_t is not None else t))
            est = estimate_eh(fv, target, elapsed, self._eh_calib,
                              load_demand_kw=load_kw)
            fam = class_family(fv)
            return select(est, family=fam,
                          severity=float(fv.get("severity_score", 0.0))).action
        except Exception:
            return None

    def step(
        self,
        t: float,
        telemetry: TelemetrySample,
        events: list[EventLog],
    ) -> tuple[Detection, list[Action]]:
        self._telemetry.append(telemetry)
        for ev in events:
            self._events.append(ev)

        if not self._should_invoke(telemetry, events):
            return Detection(asset=None, attack_class="none", confidence=0.0), []

        # Incident-elapsed tracking for the EH fallback ranker: a new incident
        # starts after >=30 s without a triggered tick.
        if self._last_trigger_t is None or (t - self._last_trigger_t) > 30.0:
            self._incident_start_t = t
        self._last_trigger_t = t

        state: AgentState = {
            "t": t,
            "telemetry_window": [_telemetry_to_dict(s) for s in self._telemetry],
            "event_window": [_event_to_dict(e) for e in self._events],
            "mode": OperationMode.SCRIPT_GEN,
            "prompt_strategy": self.prompt_strategy,
            "is_script_changed": False,
            "runtime_safe": self.runtime_safe,
        }

        # Five-agent pipeline, with ablation flags applied per stage.
        ab = self.ablation
        state = telemetry_analyst_agent(state)

        if ab == "deterministic_only":
            # Skip every LLM stage; emit a deterministic class+action from
            # the Telemetry Analyst's FeatureView.
            from ..baselines.deterministic_energy_policy.adapter import (
                _CLASS_FROM_SIGNAL, _FORCED_ACTION_BY_CLASS,
            )
            fv = state.get("feature_view", {}) or {}
            sev = float(fv.get("severity_score", 0.0))
            cls = _CLASS_FROM_SIGNAL.get(fv.get("dominant_signal", "none"),
                                            "none")
            if sev < 0.30:
                cls = "none"
            target = fv.get("dominant_asset") or ""
            action_name = _FORCED_ACTION_BY_CLASS.get(cls, "no_op")
            state.update({
                "attack_class": AttackClass(cls),
                "affected_asset": target,
                "confidence": sev,
                "rationale": f"deterministic|{fv.get('dominant_signal','none')}",
                "expected_impact_tier": ImpactTier.MEDIUM if cls != "none" else ImpactTier.NEGLIGIBLE,
                "final_action": action_name,
                "final_action_target": target,
                "final_impact_tier": "medium" if cls != "none" else "negligible",
                "coordinator_decision": "auto",
                "caution": False,
            })
        else:
            mem = None if ab == "no_incident_memory" else self._memory
            fv0 = state.get("feature_view", {}) or {}
            sig = fv0.get("dominant_signal", "none")
            use_cache = (
                self.llm_invoke_interval_s is not None
                and self._cached_llm_stage is not None
                and self._cached_signal == sig
                and self._last_llm_invoke_t is not None
                and (t - self._last_llm_invoke_t) < self.llm_invoke_interval_s
            )
            if use_cache:
                state.update(dict(self._cached_llm_stage))
                state["hypothesis_cached"] = True
            elif ab == "no_self_consistency":
                from .self_consistency import vote
                from .agents import _llm_json, _read_prompt, _fill, _strategy_hint
                template, _ = _read_prompt("hypothesis.txt")
                fv = state.get("feature_view", {}) or {}
                prompt = _fill(template,
                                feature_view=__import__("json").dumps(fv),
                                telemetry_window=__import__("json").dumps(state.get("telemetry_window", [])[-5:], default=str),
                                event_window=__import__("json").dumps(state.get("event_window", [])[-20:], default=str),
                                strategy_hint=_strategy_hint(state.get("prompt_strategy", PromptStrategy.COT)),
                                memory="(disabled)")
                aggregate = vote([_llm_json(
                    prompt, role="hypothesis",
                    sink=state.setdefault("llm_traces", []))])
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
                import uuid as _uuid
                state["decision_id"] = str(_uuid.uuid4())
            else:
                state["k_setting"] = self.k_setting
                state = hypothesis_agent(state, memory=mem,
                                         compact=self.compact_prompt)

            if ab != "no_energy_impact_agent":
                state = energy_impact_agent(state)
            else:
                # Skip the per-action impact estimator; substitute a uniform
                # placeholder so the coordinator still has the schema.
                state["impact_estimates"] = [
                    {"action": "no_op", "tier": "negligible",
                     "expected_curt_kwh": 0.0, "expected_ens_kwh": 0.0,
                     "rationale": "ablation: estimator disabled"},
                    {"action": "freeze_setpoint", "tier": "low",
                     "expected_curt_kwh": 0.05, "expected_ens_kwh": 0.0,
                     "rationale": "ablation: uniform fallback"},
                    {"action": "isolate_inverter", "tier": "high",
                     "expected_curt_kwh": 1.0, "expected_ens_kwh": 0.0,
                     "rationale": "ablation: uniform fallback"},
                    {"action": "request_ied_revalidation", "tier": "negligible",
                     "expected_curt_kwh": 0.02, "expected_ens_kwh": 0.0,
                     "rationale": "ablation: uniform fallback"},
                    {"action": "throttle_ramp", "tier": "low",
                     "expected_curt_kwh": 0.1, "expected_ens_kwh": 0.0,
                     "rationale": "ablation: uniform fallback"},
                ]

            if use_cache:
                pass  # caution verdict was cached with the hypothesis bundle
            elif ab != "no_caution_agent":
                state = caution_eval_agent(state)
            else:
                state["caution"] = False
                state["caution_reason"] = "ablation: caution disabled"

            if not use_cache and self.llm_invoke_interval_s is not None:
                self._cached_llm_stage = {
                    k: state.get(k) for k in (
                        "attack_class", "affected_asset", "confidence",
                        "rationale", "proposed_action", "proposed_action_target",
                        "expected_impact_tier", "hypothesis", "decision_id",
                        "caution", "caution_reason",
                    ) if k in state
                }
                self._cached_signal = sig
                self._last_llm_invoke_t = t

            if self.policy_mode == "safety_projection":
                from .safety_projection import project
                fv2 = state.get("feature_view", {}) or {}
                # Protocol-level (hard) evidence class hint — deliberately
                # NOT derived from z-scores/voltage alone, so benign physical
                # transients can never trigger the stand-down veto rule.
                if fv2.get("n_command_events", 0):
                    det_hint = "command_spoof"
                elif fv2.get("n_tampered_events", 0) or fv2.get("integrity_residual"):
                    det_hint = "fdi"
                elif (fv2.get("persistent_freeze")
                      or fv2.get("n_dup_events", 0) >= 2
                      or fv2.get("telemetry_stale_ticks", 0) >= 3):
                    det_hint = "dos"
                else:
                    det_hint = None
                proj = project(
                    attack_class=state.get("attack_class", AttackClass.NONE).value,
                    proposed_action=state.get("proposed_action"),
                    confidence=float(state.get("confidence", 0.0)),
                    severity=float(fv2.get("severity_score", 0.0)),
                    cautioned=bool(state.get("caution", False)),
                    parse_ok=bool(state.get("proposed_action")),
                    det_class_hint=det_hint,
                )
                if (self.fallback_estimator == "eh"
                        and proj.source == "deterministic_fallback"
                        and len(proj.safe_candidate_set) > 1):
                    eh_action = self._eh_fallback_choice(state, t)
                    if eh_action in proj.safe_candidate_set:
                        proj.executed_action = eh_action
                        proj.fallback_reason = (
                            f"{proj.fallback_reason}; EH-ranked within safe set")
                state["final_action"] = proj.executed_action
                state["final_action_target"] = (state.get("proposed_action_target")
                                                or state.get("affected_asset") or "")
                state["final_impact_tier"] = (
                    state.get("expected_impact_tier", ImpactTier.NEGLIGIBLE).value
                    if hasattr(state.get("expected_impact_tier", ImpactTier.NEGLIGIBLE), "value")
                    else "negligible")
                state["coordinator_decision"] = (
                    "hitl_required" if proj.hitl_required else "auto")
                state["coordinator_reason"] = (
                    f"safety_projection:{proj.source}")
                state["projection"] = proj.to_dict()
            # Coordinator with optional class-aware avoidance.
            elif ab == "no_class_aware_avoidance":
                # Bypass the coordinator's class-specific override + avoidance
                # set; just take the Hypothesis Agent's proposal verbatim.
                proposed = state.get("proposed_action", "no_op")
                state["final_action"] = proposed
                state["final_action_target"] = state.get("proposed_action_target", "")
                tier_obj = state.get("expected_impact_tier", ImpactTier.NEGLIGIBLE)
                state["final_impact_tier"] = tier_obj.value if hasattr(tier_obj, "value") else str(tier_obj)
                state["coordinator_decision"] = "auto"
                state["coordinator_reason"] = "ablation: no class-aware avoidance"
            else:
                state = coordinator_agent(state)

            # HITL fallback.
            if ab == "no_hitl_fallback" and state.get("coordinator_decision") == "hitl_required":
                state["coordinator_decision"] = "auto"
                state["final_action"] = state.get("chosen_for_hitl",
                                                    state.get("final_action", "no_op"))

        self._last_decision = state

        det = Detection(
            asset=state.get("affected_asset") or None,
            attack_class=state.get("attack_class", AttackClass.NONE).value,
            confidence=state.get("confidence", 0.0),
            rationale=state.get("rationale", ""),
        )

        action_name = state.get("final_action", "no_op")
        target = state.get("final_action_target", "") or state.get("affected_asset", "")
        impact_tier = state.get("final_impact_tier", "negligible")
        decision_kind = state.get("coordinator_decision", "auto")

        actions: list[Action] = []
        if (
            action_name in MITIGATIONS
            and action_name != "no_op"
            and decision_kind == "auto"
            and state.get("confidence", 0.0) >= self.min_confidence_to_act
        ):
            actions.append(Action(
                name=action_name,
                target=target or "",
                params={"rationale": state.get("rationale", ""),
                        "impact_tier": impact_tier,
                        "coordinator_reason": state.get("coordinator_reason", "")},
            ))

        # Persist this decision so subsequent invocations can retrieve
        # analogous patterns through the memory store.
        if self._memory is not None and state.get("attack_class", AttackClass.NONE) != AttackClass.NONE:
            try:
                fv = state.get("feature_view", {}) or {}
                kind = IncidentMemory.asset_kind_from_id(fv.get("dominant_asset"))
                rec = IncidentRecord(
                    decision_id=state.get("decision_id", ""),
                    signature=IncidentMemory.signature(
                        fv.get("dominant_signal", "none"), kind),
                    attack_class=state.get("attack_class", AttackClass.NONE).value,
                    affected_asset=state.get("affected_asset") or None,
                    chosen_action=action_name,
                    outcome_tier=impact_tier,
                    confidence=float(state.get("confidence", 0.0)),
                    rationale=state.get("rationale", "")[:200],
                )
                self._memory.append(rec)
            except Exception:
                pass

        return det, actions
