"""Coordinator (Critic) Agent — final action selection.

Inputs (already populated in AgentState by upstream nodes):
  - hypothesis: vote-aggregated hypothesis dict (attack_class, asset, conf, …)
  - feature_view: the deterministic FeatureView dict
  - impact_estimates: list of ImpactEstimate dicts per candidate mitigation
  - caution: bool from Caution Agent
  - caution_reason: str

Output written into the state:
  - final_action: chosen mitigation primitive (str)
  - final_action_target: chosen target asset (str)
  - final_impact_tier: ImpactTier of the chosen action
  - coordinator_decision: "auto" | "hitl_required" | "withhold"
  - coordinator_reason: explanation string

Decision policy (see paper §4.4):
  1. If severity_score < 0.3 OR attack_class == "none":  → no_op (auto)
  2. If caution flagged AND chosen action would be high/severe:
        if confidence ≥ confidence_floor → degrade to safest reversible action
        else                              → withhold and route to HITL
  3. Otherwise pick the impact-minimising mitigation that mitigates the
     dominant signal (cheapest_safe over the impact estimates).
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .energy_estimator import ImpactEstimate, cheapest_safe


_REVERSIBLE = {"freeze_setpoint", "throttle_ramp", "request_ied_revalidation", "no_op"}
_IRREVERSIBLE = {"isolate_inverter"}


def _impact_tier_for(action: str, estimates: list[dict]) -> str:
    for e in estimates:
        if e.get("action") == action:
            return e.get("tier", "negligible")
    return "negligible"


def coordinator_agent(state: dict[str, Any]) -> dict[str, Any]:
    fv = state.get("feature_view", {}) or {}
    hyp = state.get("hypothesis", {}) or {}
    impacts: list[dict] = state.get("impact_estimates", []) or []
    cautioned = bool(state.get("caution", False))
    confidence = float(hyp.get("confidence", 0.0))
    severity = float(fv.get("severity_score", 0.0))
    attack_class = hyp.get("attack_class", "none")

    target = (hyp.get("affected_asset")
              or hyp.get("proposed_action_target")
              or fv.get("dominant_asset"))

    estimates_obj = [ImpactEstimate(**e) for e in impacts]

    # Rule 1: nothing seems wrong → no_op.
    if severity < 0.3 or attack_class == "none":
        state["final_action"] = "no_op"
        state["final_action_target"] = ""
        state["final_impact_tier"] = "negligible"
        state["coordinator_decision"] = "auto"
        state["coordinator_reason"] = "severity below threshold; no mitigation needed"
        return state

    # impact-aware pick over candidates that match the dominant attack class.
    avoid = set()
    # Class-specific avoidance:
    if attack_class == "fdi":
        # FDI is telemetry-layer; isolating physically is overkill.
        avoid.add("isolate_inverter")
    if attack_class == "replay":
        avoid.add("isolate_inverter")
    chosen = cheapest_safe(estimates_obj, severity, avoid=avoid)

    # Class-specific overrides: each attack class has a canonical primitive
    # whose expected impact is bounded and whose physical semantics match
    # the attack mechanism (paper §4.4).
    overrides = {
        "command_spoof": "freeze_setpoint",
        "fdi":           "freeze_setpoint",
        "replay":        "request_ied_revalidation",
        "dos":           "request_ied_revalidation",
        "firmware":      "request_ied_revalidation",
    }
    forced = overrides.get(attack_class)
    if forced is not None:
        for e in estimates_obj:
            if e.action == forced:
                chosen = e
                break

    # Rule 2: caution gate.
    if cautioned and chosen.tier in {"high", "severe"}:
        if confidence < 0.7:
            state["final_action"] = "no_op"
            state["final_action_target"] = ""
            state["final_impact_tier"] = chosen.tier
            state["coordinator_decision"] = "hitl_required"
            state["coordinator_reason"] = (
                "cautioned + high-impact + low-confidence; routed to HITL")
            state["chosen_for_hitl"] = chosen.action
            return state
        # confident enough → degrade irreversible to its reversible analogue
        if chosen.action in _IRREVERSIBLE:
            for e in estimates_obj:
                if e.action == "freeze_setpoint":
                    chosen = e
                    break

    state["final_action"] = chosen.action
    state["final_action_target"] = target or ""
    state["final_impact_tier"] = chosen.tier
    state["coordinator_decision"] = "auto"
    state["coordinator_reason"] = (
        f"impact-aware pick: {chosen.action} ({chosen.rationale})"
    )
    return state
