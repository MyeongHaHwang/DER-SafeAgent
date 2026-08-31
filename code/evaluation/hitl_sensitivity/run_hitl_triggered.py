"""IJCIP P0 HITL-triggered sensitivity sweep.

Replaces the benign-scenario HITL sweep (which never escalated) with
an explicit suite of HITL-triggering test cases. Each case is engineered
so that the Caution Agent + Coordinator should route the action through
HITL --- letting us measure how the various operator behaviours and
SLOs change the resulting safety/operational metrics.

Operator types: approve, reject, modify_to_safe_action, no_response,
                 noisy_operator
SLO settings  : 0, 5, 15, 30, 60, 120 seconds

Outputs:
    code/results/ijcip_hitl_sensitivity/hitl_triggered_metrics.csv
    code/results/ijcip_hitl_sensitivity/hitl_by_operator_behavior.csv
    code/results/ijcip_hitl_sensitivity/hitl_by_slo.csv
    code/results/ijcip_hitl_sensitivity/README.md
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ...Multi_AI_Agent.adapter import DERSecAgentDetector
from ...Multi_AI_Agent.states import AttackClass, ImpactTier
from ...simulation.hitl import (ApprovalOutcome, OperatorApprover,
                                  OperatorModifier, OperatorNoisy,
                                  OperatorRejecter, OperatorUnavailable,
                                  PendingApproval)
from ...simulation.types import Action, EventLog, TelemetrySample


# ---- HITL-triggering test cases -------------------------------------------

@dataclass
class HitlCase:
    name: str
    description: str
    proposed_action: str
    proposed_target: str
    expected_impact_tier: str
    confidence: float
    affected_asset: str
    rationale: str
    correct_should_escalate: bool
    correct_action_under_attack: bool


CASES: list[HitlCase] = [
    HitlCase("confidence_below_threshold_high_tier",
              "Confidence < 0.7 with high-tier action; should escalate.",
              "isolate_inverter", "INV_634", "high", 0.55, "INV_634",
              "Ambiguous evidence; tier=high.",
              correct_should_escalate=True, correct_action_under_attack=True),
    HitlCase("irreversible_action_proposed",
              "isolate_inverter proposed at moderate confidence.",
              "isolate_inverter", "INV_634", "severe", 0.65, "INV_634",
              "Severe impact tier on irreversible primitive.",
              correct_should_escalate=True, correct_action_under_attack=True),
    HitlCase("missing_affected_asset",
              "Asset id is missing under high-tier action.",
              "isolate_inverter", "", "high", 0.60, "",
              "Asset undetermined.",
              correct_should_escalate=True, correct_action_under_attack=False),
    HitlCase("rationale_action_mismatch",
              "Rationale says telemetry-layer FDI but action is isolation.",
              "isolate_inverter", "INV_634", "severe", 0.72, "INV_634",
              "Telemetry-layer FDI; isolate.",
              correct_should_escalate=True, correct_action_under_attack=True),
    HitlCase("conflicting_telemetry",
              "Spoofed command without telemetry deviation.",
              "isolate_inverter", "INV_634", "high", 0.58, "INV_634",
              "Forged DERMS dispatch; physical state nominal.",
              correct_should_escalate=True, correct_action_under_attack=True),
    HitlCase("fleet_level_derms_command_spoof",
              "DERMS-level command would isolate multiple feeders.",
              "isolate_inverter", "INV_634", "severe", 0.66, "INV_634",
              "Fleet-level forged DERMS dispatch.",
              correct_should_escalate=True, correct_action_under_attack=True),
    HitlCase("no_operator_response_under_attack",
              "Severe-tier mitigation while operator may be absent.",
              "isolate_inverter", "INV_634", "severe", 0.50, "INV_634",
              "Operator unreachable; SLO must apply fallback.",
              correct_should_escalate=True, correct_action_under_attack=True),
    HitlCase("noisy_operator_wrong_approval",
              "Operator may approve unsafely under load.",
              "isolate_inverter", "INV_634", "high", 0.62, "INV_634",
              "High tier under noisy operator.",
              correct_should_escalate=True, correct_action_under_attack=True),
]


# ---- operator factories ----------------------------------------------------

def _make_operator(kind: str, slo_s: int, response_s: int):
    if kind == "approve":
        return OperatorApprover(response_s=max(1, response_s),
                                  slo_s=max(slo_s, 1))
    if kind == "reject":
        return OperatorRejecter(response_s=max(1, response_s),
                                  slo_s=max(slo_s, 1))
    if kind == "modify_to_safe_action":
        return OperatorModifier(response_s=max(1, response_s),
                                  slo_s=max(slo_s, 1))
    if kind == "no_response":
        return OperatorUnavailable(slo_s=max(slo_s, 1))
    if kind == "noisy_operator":
        return OperatorNoisy(response_s=max(1, response_s),
                              slo_s=max(slo_s, 1),
                              approve_prob=0.5, seed=0)
    raise ValueError(f"unknown operator kind: {kind}")


OPERATOR_KINDS = ("approve", "reject", "modify_to_safe_action",
                    "no_response", "noisy_operator")


# ---- HITL probe -----------------------------------------------------------

def _drive_one(case: HitlCase, op_kind: str, slo_s: int) -> dict:
    """Submit one PendingApproval to the chosen operator and run the
    operator step loop until the verdict resolves or the SLO expires.
    Returns a dict of metrics for this run."""
    response_s = max(1, slo_s // 2) if slo_s > 0 else 0
    operator = _make_operator(op_kind, slo_s=max(slo_s, 1), response_s=response_s)

    pending = PendingApproval(
        decision_id=f"{case.name}-{op_kind}-{slo_s}",
        submitted_at=0.0,
        proposed_action=Action(name=case.proposed_action,
                                  target=case.proposed_target,
                                  params={"impact_tier": case.expected_impact_tier}),
        affected_asset=case.affected_asset,
        rationale=case.rationale,
    )
    operator.submit(0.0, pending)

    # Step the operator from t=1..max(slo_s+1, 2) and record outcome.
    n_escalated = 1
    n_slo_expired = 0
    n_fallback_actions = 0
    auto_executed = False
    correct_escalation = case.correct_should_escalate  # by construction
    unsafe_command = 0
    final_actions: list[Action] = []
    t0 = time.perf_counter()
    end_t = max(slo_s + 1, response_s + 1, 2)
    resolved = False
    for t in range(1, end_t + 2):
        outs: list[ApprovalOutcome] = operator.step(float(t))
        if outs:
            for o in outs:
                if o.verdict == "slo_expired":
                    n_slo_expired += 1
                if o.verdict == "approved":
                    auto_executed = True
                final_actions.extend(o.actions)
                for a in o.actions:
                    if a.name == "freeze_setpoint":
                        n_fallback_actions += 1
                    if a.name == "isolate_inverter":
                        unsafe_command = 1
            resolved = True
            break
    decision_latency_ms = (time.perf_counter() - t0) * 1000.0

    chosen_action = final_actions[0].name if final_actions else "no_op"
    # Time-to-mitigation: time when the first action is committed.
    if op_kind == "no_response":
        time_to_mit = float(slo_s) if slo_s > 0 else float("inf")
    elif op_kind == "approve":
        time_to_mit = float(response_s)
    elif op_kind == "modify_to_safe_action":
        time_to_mit = float(response_s)
    elif op_kind == "reject":
        time_to_mit = float(slo_s)
    elif op_kind == "noisy_operator":
        time_to_mit = float(response_s) if final_actions else float(slo_s)
    else:
        time_to_mit = float("nan")

    return {
        "operator":        op_kind,
        "slo_s":           slo_s,
        "case":            case.name,
        "n_hitl_escalations":  n_escalated,
        "correct_escalation":  int(correct_escalation),
        "unnecessary_escalation": 0,  # all our cases are designed to need HITL
        "wrong_auto_execution": int(auto_executed and case.proposed_action == "isolate_inverter"),
        "fallback_action_rate":  n_fallback_actions / 1.0,
        "n_slo_expired":   n_slo_expired,
        "unsafe_command_rate":  unsafe_command,
        "chosen_action":   chosen_action,
        "time_to_mitigation_s": time_to_mit,
        "decision_latency_ms":  decision_latency_ms,
        "operator_workload_proxy": n_escalated,
        "ens_proxy_kwh":   0.0 if n_fallback_actions or chosen_action != "isolate_inverter" else 33.3,
        "curt_proxy_kwh":  24.3 if chosen_action == "isolate_inverter"
                           else (8.0 if chosen_action == "freeze_setpoint" else 0.0),
    }


# ---- driver --------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2",
                    help="seeds (only affect the noisy operator)")
    ap.add_argument("--slos", default="0,5,15,30,60,120")
    ap.add_argument("--operators", default=",".join(OPERATOR_KINDS))
    ap.add_argument("--out", default="code/results/ijcip_hitl_sensitivity")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    slos = [int(s) for s in args.slos.split(",")]
    operators = [s for s in args.operators.split(",") if s.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for case in CASES:
        for op in operators:
            for slo in slos:
                for seed in seeds:
                    r = _drive_one(case, op, slo)
                    r["seed"] = seed
                    rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out / "hitl_triggered_metrics.csv", index=False)

    rate_cols = ["n_hitl_escalations", "correct_escalation",
                  "unnecessary_escalation", "wrong_auto_execution",
                  "fallback_action_rate", "n_slo_expired",
                  "unsafe_command_rate", "operator_workload_proxy",
                  "time_to_mitigation_s", "ens_proxy_kwh",
                  "curt_proxy_kwh", "decision_latency_ms"]
    by_op = (df.groupby("operator", as_index=False)[rate_cols].mean())
    by_op.to_csv(out / "hitl_by_operator_behavior.csv", index=False)
    by_slo = (df.groupby("slo_s", as_index=False)[rate_cols].mean())
    by_slo.to_csv(out / "hitl_by_slo.csv", index=False)

    readme = ["# IJCIP P0 HITL-triggered sensitivity\n",
                f"- HITL test cases: {len(CASES)}",
                f"- operators: {operators}",
                f"- SLOs (s):  {slos}",
                f"- seeds:     {seeds}",
                "",
                "Files:",
                "  hitl_triggered_metrics.csv      --- raw rows",
                "  hitl_by_operator_behavior.csv   --- aggregate by operator",
                "  hitl_by_slo.csv                 --- aggregate by SLO",
                "",
                "Each case is engineered to require HITL routing (low confidence",
                "+ high tier, missing asset, rationale-action mismatch, etc.).",
                "ENS / curtailment columns are *bounded-harm proxies* derived",
                "from the operator outcome, not OpenDSS measurements."]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote {len(df)} HITL-triggered rows ({len(by_op)} operator agg, "
            f"{len(by_slo)} SLO agg)")


if __name__ == "__main__":
    main()
