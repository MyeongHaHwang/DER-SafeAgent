"""Sweep the HITL configuration --- SLO × operator behaviour ---
across the standard scenarios.

Operators evaluated:
    approve              -- OperatorApprover
    reject               -- OperatorRejecter
    modify_to_safe_action-- OperatorModifier
    no_response          -- OperatorUnavailable
    noisy_operator       -- OperatorNoisy(approve_prob=0.5)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..caution_metrics import aggregate_run as caution_aggregate
from ..physical_curves import sweep
from ...Multi_AI_Agent.adapter import DERSecAgentDetector
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario
from ...simulation.hitl import (OperatorApprover, OperatorModifier,
                                  OperatorNoisy, OperatorRejecter,
                                  OperatorUnavailable)


SCENARIOS = [
    "code/simulation/scenarios/ieee13_fdi_inverter/config.yaml",
    "code/simulation/scenarios/ieee13_command_spoof/config.yaml",
    "code/simulation/scenarios/ieee34_command_spoof_derms/config.yaml",
]

OPERATOR_FACTORIES = {
    "approve":               lambda response_s, slo_s: OperatorApprover(response_s=response_s, slo_s=slo_s),
    "reject":                lambda response_s, slo_s: OperatorRejecter(response_s=response_s, slo_s=slo_s),
    "modify_to_safe_action": lambda response_s, slo_s: OperatorModifier(response_s=response_s, slo_s=slo_s),
    "no_response":           lambda response_s, slo_s: OperatorUnavailable(slo_s=slo_s),
    "noisy_operator":        lambda response_s, slo_s: OperatorNoisy(
                                  response_s=response_s, slo_s=slo_s,
                                  approve_prob=0.5, seed=0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--slos", default="0,5,15,30,60,120",
                    help="comma-separated SLO seconds; the response time is "
                         "set to half the SLO")
    ap.add_argument("--out", default="code/results/ijcip_hitl_sensitivity")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    slos = [int(s) for s in args.slos.split(",")]
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    runs_root = out_root / "runs"

    for scenario_path in SCENARIOS:
        cfg = yaml.safe_load(Path(scenario_path).read_text())
        for op_name, op_factory in OPERATOR_FACTORIES.items():
            for slo in slos:
                response_s = max(1, slo // 2) if slo > 0 else 0
                for seed in seeds:
                    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                          ders=cfg["ders"])
                    detector = DERSecAgentDetector(
                        name=f"der_secagent_hitl_{op_name}_slo{slo}",
                    )
                    operator = op_factory(response_s, slo if slo > 0 else 1)
                    run_scenario(scenario_path, detector, seed,
                                  out_root=str(runs_root),
                                  feeder=feeder, operator=operator)

    # Aggregate
    thrs = np.linspace(0.0, 1.0, 11)
    frames = [sweep(m.parent, thrs) for m in runs_root.rglob("manifest.json")]
    if not frames:
        print("no runs found"); return

    df = pd.concat(frames, ignore_index=True)
    head = df[df["threshold"].between(0.49, 0.51)]

    rows = []
    for (scn, det, seed), g in head.groupby(["scenario", "detector", "seed"]):
        det_str = det
        # detector encodes "der_secagent_hitl_<op>_slo<slo>"
        try:
            tail = det_str.split("hitl_", 1)[1]
            op_name, slo_part = tail.rsplit("_slo", 1)
            slo_val = int(slo_part)
        except Exception:
            op_name, slo_val = "approve", -1
        # HITL log
        run_dir = next(runs_root.rglob(f"{det}/seed{seed}"))
        hitl_path = run_dir / "hitl.jsonl"
        n_hitl = 0; n_slo_expired = 0; n_fallback = 0
        if hitl_path.exists():
            for line in hitl_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_hitl += 1
                if rec.get("verdict") == "slo_expired":
                    n_slo_expired += 1
                if any(a.get("name") == "freeze_setpoint" for a in rec.get("actions", [])):
                    n_fallback += 1
        rows.append({
            "scenario": scn, "operator": op_name, "slo_s": slo_val,
            "seed": seed,
            "ens_kwh": float(g["ens_kwh"].mean()),
            "curt_kwh": float(g["curt_kwh"].mean()),
            "voltage_frac": float(g["voltage_frac"].mean()),
            "freq_dev_hz": float(g["freq_dev_hz"].mean()),
            "n_hitl_escalations": n_hitl,
            "n_slo_expired": n_slo_expired,
            "n_safe_fallback_actions": n_fallback,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_root / "hitl_sensitivity.csv", index=False)
    agg = (out_df.groupby(["scenario", "operator", "slo_s"], as_index=False)
                  [["ens_kwh", "curt_kwh", "n_hitl_escalations",
                     "n_slo_expired", "n_safe_fallback_actions"]].mean())
    agg.to_csv(out_root / "hitl_sensitivity_agg.csv", index=False)

    readme = ["# HITL sensitivity sweep\n",
                f"- scenarios: {len(SCENARIOS)}",
                f"- operators: {', '.join(OPERATOR_FACTORIES.keys())}",
                f"- SLOs (s): {slos}",
                f"- seeds: {seeds}",
                "",
                "Files:",
                "  hitl_sensitivity.csv     --- one row per (scenario, operator, slo, seed)",
                "  hitl_sensitivity_agg.csv --- mean over seeds"]
    (out_root / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote hitl_sensitivity rows: {len(out_df)}; agg rows: {len(agg)}")


if __name__ == "__main__":
    main()
