"""IJCIP P0 objective-sensitivity sweep on the trade-off scenario.

Unlike the original sweep --- which collapses on benign scenarios
because the Coordinator's class-aware override dominates the objective
weights --- this driver runs the sweep on the engineered
``objective_tradeoff_case`` from
``code/simulation/scenarios/ijcip_sensitive/`` plus a synthetic
candidate set in which non-equivalent actions actually exist. The
objective ``cheapest_safe`` function is parameterised over
``ens_weight``, ``tier_penalty``, and ``severity_threshold`` so we can
score the *intended* sensitivity directly, then label each grid point
with a risk-posture profile.

Outputs:
    code/results/ijcip_objective_sensitivity/objective_grid_expanded.csv
    code/results/ijcip_objective_sensitivity/pareto_points_expanded.csv
    code/results/ijcip_objective_sensitivity/risk_posture_summary.csv
    code/results/ijcip_objective_sensitivity/README.md
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ...Multi_AI_Agent.energy_estimator import (CANDIDATE_ACTIONS,
                                                  ImpactEstimate,
                                                  cheapest_safe)


# --------- synthetic candidate set with a real trade-off ---------

def _candidate_set() -> list[ImpactEstimate]:
    """A non-degenerate candidate set in which the Coordinator's
    argmin actually changes with the weights.

    The (curt, ENS) trade-off is engineered so that:
    * curtailment_minimizing weights (low ens_weight) prefer
      throttle_ramp (tiny curt; partial ENS coverage);
    * balanced weights prefer freeze_setpoint;
    * reliability_first weights (high ens_weight) prefer
      isolate_inverter (zero ENS at the cost of large curtailment).
    Tier penalty controls when isolate_inverter is locked out.
    """
    return [
        # no_op: zero curtailment, but real ENS exposure
        ImpactEstimate(action="no_op", tier="medium",
                        expected_curt_kwh=0.0,
                        expected_ens_kwh=2.5,
                        rationale="passes through attack"),
        # freeze_setpoint: balanced trade-off
        ImpactEstimate(action="freeze_setpoint", tier="low",
                        expected_curt_kwh=2.0,
                        expected_ens_kwh=0.6,
                        rationale="reverts to last-validated setpoint"),
        # throttle_ramp: smallest curtailment, residual ENS
        ImpactEstimate(action="throttle_ramp", tier="low",
                        expected_curt_kwh=0.3,
                        expected_ens_kwh=1.4,
                        rationale="caps ramp; partial mitigation"),
        # request_ied_revalidation: tiny curtailment, residual ENS
        ImpactEstimate(action="request_ied_revalidation", tier="negligible",
                        expected_curt_kwh=0.1,
                        expected_ens_kwh=1.6,
                        rationale="brief telemetry gap; partial coverage"),
        # isolate_inverter: highest curtailment, zero ENS
        ImpactEstimate(action="isolate_inverter", tier="high",
                        expected_curt_kwh=4.0,
                        expected_ens_kwh=0.0,
                        rationale="full curtailment of asset; prevents ENS"),
    ]


def _classify_posture(action: str) -> str:
    if action == "isolate_inverter":   return "safety_conservative"
    if action == "freeze_setpoint":    return "balanced"
    if action == "throttle_ramp":      return "curtailment_minimizing"
    if action == "request_ied_revalidation": return "reliability_first"
    if action == "no_op":              return "reliability_first"
    return "balanced"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens-weights",   default="1,2,5,10,20")
    ap.add_argument("--tier-penalties", default="0,1.5,3,5")
    ap.add_argument("--confidence-thresholds", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--severity",      type=float, default=0.85,
                    help="severity_score injected into the optimiser")
    ap.add_argument("--out",
                    default="code/results/ijcip_objective_sensitivity")
    args = ap.parse_args()

    ens_weights = [float(s) for s in args.ens_weights.split(",")]
    tier_penalties = [float(s) for s in args.tier_penalties.split(",")]
    conf_thresholds = [float(s) for s in args.confidence_thresholds.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cset = _candidate_set()
    grid_rows = []

    for ens_w, tier_p, conf_t in itertools.product(ens_weights,
                                                      tier_penalties,
                                                      conf_thresholds):
        chosen = cheapest_safe(cset, severity_score=args.severity,
                                  ens_weight=ens_w, tier_penalty=tier_p)
        # If confidence floor would gate the action, that effectively
        # forces an HITL path (we don't have a Caution Agent here, so we
        # simply note when a choice would have been blocked under the
        # floor with no high-tier rerouting).
        gated = (chosen.tier in {"high", "severe"}) and (args.severity < conf_t)
        grid_rows.append({
            "ens_weight":      ens_w,
            "tier_penalty":    tier_p,
            "conf_threshold":  conf_t,
            "selected_action": chosen.action,
            "selected_tier":   chosen.tier,
            "expected_curt_kwh": chosen.expected_curt_kwh,
            "expected_ens_kwh":  chosen.expected_ens_kwh,
            "hitl_required":   bool(gated),
            "unsafe_command":  int(chosen.tier in {"high", "severe"}
                                    and not gated),
            "risk_posture":    _classify_posture(chosen.action),
        })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out / "objective_grid_expanded.csv", index=False)

    # Pareto on (curtailment, ENS).
    pareto = []
    g = grid_df.dropna(subset=["expected_curt_kwh", "expected_ens_kwh"]) \
                .reset_index(drop=True)
    for i, r in g.iterrows():
        dominated = False
        for j, o in g.iterrows():
            if i == j: continue
            if (o["expected_curt_kwh"] <= r["expected_curt_kwh"] and
                o["expected_ens_kwh"]  <= r["expected_ens_kwh"]  and
                (o["expected_curt_kwh"] < r["expected_curt_kwh"] or
                 o["expected_ens_kwh"]  < r["expected_ens_kwh"])):
                dominated = True; break
        if not dominated:
            pareto.append(dict(r))
    pd.DataFrame(pareto).to_csv(out / "pareto_points_expanded.csv", index=False)

    # Risk-posture summary: how often each posture wins as we move
    # through the grid.
    rp = (grid_df.groupby("risk_posture", as_index=False)
                  .size().rename(columns={"size": "n_grid_points"}))
    rp["frac_grid"] = rp["n_grid_points"] / len(grid_df)
    rp.to_csv(out / "risk_posture_summary.csv", index=False)

    readme = ["# IJCIP P0 objective sensitivity (trade-off scenario)\n",
                f"- ens_weights:   {ens_weights}",
                f"- tier_penalty:  {tier_penalties}",
                f"- conf_threshold:{conf_thresholds}",
                f"- severity:      {args.severity}",
                "",
                "Files:",
                "  objective_grid_expanded.csv --- per-grid-point selection",
                "  pareto_points_expanded.csv  --- Pareto-optimal grid points",
                "  risk_posture_summary.csv    --- frequency of each risk posture",
                "",
                "The candidate set is engineered so the Coordinator's argmin",
                "actually changes with the weights (no_op trades curtailment",
                "for ENS; isolate_inverter trades ENS for tier penalty).",
                "Risk postures: reliability_first / curtailment_minimizing /",
                "balanced / safety_conservative."]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote objective grid: {len(grid_df)} points, "
            f"pareto={len(pareto)}")


if __name__ == "__main__":
    main()
