"""Sweep the Coordinator objective weights over a grid and re-run the
3-scenario × N-seed matrix for each grid point.

The Coordinator objective is::
    J(a) = curtailment_kWh + ens_weight * ENS_kWh
         + tier_penalty * 1{tier(a) in {high, severe}}

Sweep:
    ens_weight    in {1, 2, 5, 10, 20}
    tier_penalty  in {0, 1.5, 3, 5}
    confidence_threshold (min_confidence_to_act) in {0.3, 0.5, 0.7, 0.9}

For each grid point we capture mean ENS, mean curtailment, the unsafe-
command rate, and the selected-action distribution.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..caution_metrics import aggregate_run as caution_aggregate
from ..physical_curves import sweep
from ...Multi_AI_Agent.adapter import DERSecAgentDetector
from ...Multi_AI_Agent import energy_estimator as _ee
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario


SCENARIOS = [
    "code/simulation/scenarios/ieee13_fdi_inverter/config.yaml",
    "code/simulation/scenarios/ieee13_command_spoof/config.yaml",
    "code/simulation/scenarios/ieee34_command_spoof_derms/config.yaml",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--ens-weights", default="1,2,5,10,20")
    ap.add_argument("--tier-penalties", default="0,1.5,3,5")
    ap.add_argument("--confidence-thresholds", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--out", default="code/results/ijcip_objective_sensitivity")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    ens_weights = [float(s) for s in args.ens_weights.split(",")]
    tier_penalties = [float(s) for s in args.tier_penalties.split(",")]
    conf_thresholds = [float(s) for s in args.confidence_thresholds.split(",")]
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    runs_root = out_root / "runs"

    grid_rows = []

    for ens_w, tier_p, conf_t in itertools.product(ens_weights, tier_penalties,
                                                    conf_thresholds):
        # Patch the energy_estimator default weights for the duration of
        # this grid point. cheapest_safe takes weights as keyword args
        # already; we override the default by binding kwargs in the
        # coordinator's call site through a closure on _ee.cheapest_safe.
        original_fn = _ee.cheapest_safe
        def patched(estimates, severity_score, avoid=None,
                     ens_weight=ens_w, tier_penalty=tier_p,
                     severity_threshold=0.30):
            return original_fn(estimates, severity_score, avoid=avoid,
                                  ens_weight=ens_weight, tier_penalty=tier_penalty,
                                  severity_threshold=severity_threshold)
        _ee.cheapest_safe = patched
        # Ensure code.Multi_AI_Agent.coordinator picks up the patched fn.
        import code.Multi_AI_Agent.coordinator as _coord
        _coord.cheapest_safe = patched

        try:
            grid_label = f"ens{ens_w}_tier{tier_p}_conf{conf_t}"
            for scenario_path in SCENARIOS:
                cfg = yaml.safe_load(Path(scenario_path).read_text())
                for seed in seeds:
                    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                          ders=cfg["ders"])
                    detector = DERSecAgentDetector(
                        name=f"der_secagent_{grid_label}",
                        min_confidence_to_act=conf_t,
                    )
                    run_scenario(scenario_path, detector, seed,
                                  out_root=str(runs_root / grid_label),
                                  feeder=feeder)
        finally:
            _ee.cheapest_safe = original_fn
            _coord.cheapest_safe = original_fn

        # Aggregate for this grid point.
        thrs = np.linspace(0.0, 1.0, 11)
        frames = [sweep(m.parent, thrs) for m in (runs_root / grid_label).rglob("manifest.json")]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        head = df[df["threshold"].between(0.49, 0.51)]
        cau_rows = [caution_aggregate(m.parent) for m in (runs_root / grid_label).rglob("manifest.json")]
        cau = pd.DataFrame(cau_rows) if cau_rows else pd.DataFrame()
        # selected-action distribution
        n_iso = n_freeze = n_revalid = n_no_op = n_throttle = 0
        for m in (runs_root / grid_label).rglob("decisions.jsonl"):
            for line in m.read_text().splitlines():
                if not line.strip():
                    continue
                import json as _json
                rec = _json.loads(line)
                for a in rec.get("actions", []):
                    name = a.get("name", "no_op")
                    if name == "isolate_inverter":           n_iso += 1
                    elif name == "freeze_setpoint":          n_freeze += 1
                    elif name == "request_ied_revalidation": n_revalid += 1
                    elif name == "throttle_ramp":            n_throttle += 1
                    else:                                      n_no_op += 1
        n_actions = max(n_iso + n_freeze + n_revalid + n_throttle + n_no_op, 1)
        grid_rows.append({
            "ens_weight":      ens_w,
            "tier_penalty":    tier_p,
            "conf_threshold":  conf_t,
            "mean_ens_kwh":    float(head["ens_kwh"].mean()) if not head.empty else float("nan"),
            "mean_curt_kwh":   float(head["curt_kwh"].mean()) if not head.empty else float("nan"),
            "mean_voltage_frac": float(head["voltage_frac"].mean()) if not head.empty else float("nan"),
            "unsafe_command_rate": float(cau["unsafe_action_rate"].mean()) if not cau.empty else 0.0,
            "frac_isolate":     n_iso / n_actions,
            "frac_freeze":      n_freeze / n_actions,
            "frac_revalidate":  n_revalid / n_actions,
            "frac_throttle":    n_throttle / n_actions,
            "frac_no_op":       n_no_op / n_actions,
        })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out_root / "objective_grid.csv", index=False)

    # Pareto frontier: (mean_curt_kwh, mean_ens_kwh) --- lower is better.
    pareto = []
    grid = grid_df.dropna(subset=["mean_curt_kwh", "mean_ens_kwh"]).reset_index(drop=True)
    for i, row in grid.iterrows():
        dominated = False
        for j, other in grid.iterrows():
            if i == j:
                continue
            if (other["mean_curt_kwh"] <= row["mean_curt_kwh"] and
                other["mean_ens_kwh"]  <= row["mean_ens_kwh"]  and
                (other["mean_curt_kwh"] < row["mean_curt_kwh"] or
                 other["mean_ens_kwh"]  < row["mean_ens_kwh"])):
                dominated = True; break
        if not dominated:
            pareto.append(dict(row))
    pd.DataFrame(pareto).to_csv(out_root / "pareto_points.csv", index=False)

    readme = ["# Objective sensitivity / Pareto\n",
                f"- ens_weights:    {ens_weights}",
                f"- tier_penalties: {tier_penalties}",
                f"- conf_thresholds: {conf_thresholds}",
                f"- seeds: {seeds}",
                "",
                "Files:",
                "  objective_grid.csv  --- one row per (ens_w, tier_p, conf_t) grid point",
                "  pareto_points.csv   --- Pareto-optimal grid points on (curtailment, ENS)"]
    (out_root / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote objective_grid with {len(grid_df)} grid points; "
          f"pareto frontier has {len(pareto)} points")


if __name__ == "__main__":
    main()
