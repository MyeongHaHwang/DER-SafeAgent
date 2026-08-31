"""Ablation driver --- evaluates DER-SecAgent under each ablation flag on
the standard 3-scenario × 5-seed matrix and emits one ablation_metrics
CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .caution_metrics import aggregate_run as caution_aggregate
from .detection_metrics import aggregate_run as detection_aggregate, macro_f1
from .physical_curves import sweep
from ..Multi_AI_Agent.adapter import ABLATION_FLAGS, DERSecAgentDetector
from ..simulation.feeder import StubFeeder
from ..simulation.harness import run_scenario


SCENARIOS = [
    "code/simulation/scenarios/ieee13_fdi_inverter/config.yaml",
    "code/simulation/scenarios/ieee13_command_spoof/config.yaml",
    "code/simulation/scenarios/ieee34_command_spoof_derms/config.yaml",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="code/results/ijcip_ablation")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    runs_root = out_root / "runs"

    for scenario_path in SCENARIOS:
        cfg = yaml.safe_load(Path(scenario_path).read_text())
        for ab in ABLATION_FLAGS:
            for seed in seeds:
                feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                      ders=cfg["ders"])
                detector = DERSecAgentDetector(
                    name=f"der_secagent_{ab}", ablation=ab,
                )
                run_scenario(scenario_path, detector, seed,
                              out_root=str(runs_root), feeder=feeder)

    # Aggregate
    thrs = np.linspace(0.0, 1.0, 11)
    frames = [sweep(m.parent, thrs) for m in runs_root.rglob("manifest.json")]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df.to_csv(out_root / "physical_metrics.csv", index=False)
        head = df[df["threshold"].between(0.49, 0.51)]
        agg = (head.groupby(["scenario", "detector"], as_index=False)
                  [["ens_kwh", "curt_kwh", "voltage_frac", "freq_dev_hz"]]
                  .mean())
    else:
        agg = pd.DataFrame()

    cau_rows = [caution_aggregate(m.parent) for m in runs_root.rglob("manifest.json")]
    cau_df = pd.DataFrame(cau_rows) if cau_rows else pd.DataFrame()
    if not cau_df.empty:
        cau_agg = (cau_df.groupby(["scenario", "detector"], as_index=False)
                    [["fp_action_rate", "fn_action_rate", "unsafe_action_rate"]].mean())
    else:
        cau_agg = pd.DataFrame()

    det_frames = [detection_aggregate(m.parent) for m in runs_root.rglob("manifest.json")]
    if det_frames:
        det_df = pd.concat(det_frames, ignore_index=True)
        rows = []
        for (scn, detn, sd), g in det_df.groupby(["scenario", "detector", "seed"]):
            rows.append({"scenario": scn, "detector": detn, "seed": sd,
                          "macro_f1": macro_f1(g)})
        f1_df = pd.DataFrame(rows)
        f1_agg = (f1_df.groupby(["scenario", "detector"], as_index=False)
                    ["macro_f1"].mean())
    else:
        f1_agg = pd.DataFrame()

    if not agg.empty:
        agg["ablation"] = agg["detector"].str.replace("der_secagent_", "", regex=False)
        if not cau_agg.empty:
            cau_agg["ablation"] = cau_agg["detector"].str.replace("der_secagent_", "", regex=False)
            agg = agg.merge(cau_agg.drop(columns=["detector"]),
                              on=["scenario", "ablation"], how="left")
        if not f1_agg.empty:
            f1_agg["ablation"] = f1_agg["detector"].str.replace("der_secagent_", "", regex=False)
            agg = agg.merge(f1_agg.drop(columns=["detector"]),
                              on=["scenario", "ablation"], how="left")
        agg.to_csv(out_root / "ablation_metrics.csv", index=False)

    # Adversarial-suite scoring per ablation: we re-use the run_robustness
    # evaluator with each ablation as the detector instance and aggregate
    # over the six perturbation families.
    adv_rows = []
    try:
        from .adversarial_safety.perturbations import GENERATORS, build_suite
        from .adversarial_safety.run_robustness import (_evaluate_violation,
                                                           _step_detector)
        cases = build_suite(seed=0, n_per_family=5)
        for ab in ABLATION_FLAGS:
            for fam in GENERATORS:
                sub = [c for c in cases if c.perturbation == fam]
                detector = DERSecAgentDetector(name=f"der_secagent_{ab}",
                                                  ablation=ab)
                agg = {"policy_violation": 0, "forbidden_action": 0,
                        "schema_failure": 0, "safe_fallback": 0,
                        "hitl_escalation": 0, "abstained": 0,
                        "correct_refusal": 0, "recovery_success": 0,
                        "unauthorized_command": 0}
                for c in sub:
                    rec = _step_detector(detector, c)
                    flags = _evaluate_violation(c, rec["action"], rec["tier"],
                                                  rec["hitl"], rec["schema_ok"])
                    for k, v in flags.items():
                        agg[k] += v
                n = max(len(sub), 1)
                adv_rows.append({"ablation": ab, "perturbation": fam,
                                  "n_cases": len(sub),
                                  **{f"{k}_rate": agg[k] / n for k in agg}})
        if adv_rows:
            pd.DataFrame(adv_rows).to_csv(
                out_root / "ablation_adversarial_metrics.csv", index=False)
    except Exception as exc:
        print("adversarial ablation skipped:", repr(exc)[:200])

    readme = ["# DER-SecAgent ablation matrix\n",
                f"Flags evaluated: {', '.join(ABLATION_FLAGS)}",
                "",
                "## Files",
                "- `ablation_metrics.csv` --- physical-impact + caution + macro_f1 per (scenario, ablation)",
                "- `ablation_adversarial_metrics.csv` --- robustness rates per (ablation, perturbation family)",
                "",
                "Each ablation row averages over 5 seeds × 3 scenarios.",
                "Adversarial rows use 5 cases per perturbation family from the",
                "deterministic suite at seed=0."]
    (out_root / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote ablation_metrics with {len(ABLATION_FLAGS)} flags × 3 scenarios "
          f"and {len(adv_rows)} adversarial rows")


if __name__ == "__main__":
    main()
