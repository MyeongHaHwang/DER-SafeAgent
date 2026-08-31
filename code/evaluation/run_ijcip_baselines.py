"""Run all baselines (existing + new IJCIP additions) on the standard
co-simulation matrix and emit a single comparison CSV.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .caution_metrics import aggregate_run as caution_aggregate
from .detection_metrics import aggregate_run as detection_aggregate, macro_f1
from .physical_curves import sweep
from ..simulation.feeder import StubFeeder
from ..simulation.harness import run_scenario


SCENARIOS = [
    "code/simulation/scenarios/ieee13_fdi_inverter/config.yaml",
    "code/simulation/scenarios/ieee13_command_spoof/config.yaml",
    "code/simulation/scenarios/ieee34_command_spoof_derms/config.yaml",
]

DETECTORS: dict[str, str] = {
    "rule_ids":                    "code.baselines.rule_ids.adapter:RuleIDS",
    "single_llm":                  "code.baselines.single_llm.runner:SingleLLM",
    "prior_mas":                   "code.baselines.prior_mas.adapter:PriorMAS",
    # IJCIP additions
    "safe_single_llm":             "code.baselines.safe_single_llm.runner:SafeSingleLLM",
    "deterministic_energy_policy": "code.baselines.deterministic_energy_policy.adapter:DeterministicEnergyPolicy",
    "single_llm_with_caution":     "code.baselines.single_llm_with_caution.runner:SingleLLMWithCaution",
    "prior_mas_with_safety":       "code.baselines.prior_mas_with_safety.adapter:PriorMASWithSafety",
    "oracle_class_policy":         "code.baselines.oracle_class_policy.adapter:OracleClassPolicy",
    "der_secagent":                "code.Multi_AI_Agent.adapter:DERSecAgentDetector",
}


def _instantiate(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def _configure_oracle(detector, scenario_cfg: dict) -> None:
    if detector.name != "oracle_class_policy":
        return
    gt = scenario_cfg.get("ground_truth", {})
    detector.configure(
        attack_class=gt.get("attack_class", "none"),
        target=gt.get("affected_asset", ""),
        start_s=float(gt.get("start_s", 0.0)),
        end_s=float(gt.get("end_s", 0.0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="code/results/ijcip_baselines")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    runs_root = out_root / "runs"

    for scenario_path in SCENARIOS:
        cfg = yaml.safe_load(Path(scenario_path).read_text())
        for det_name, det_spec in DETECTORS.items():
            for seed in seeds:
                feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                      ders=cfg["ders"])
                detector = _instantiate(det_spec)
                _configure_oracle(detector, cfg)
                run_scenario(scenario_path, detector, seed,
                              out_root=str(runs_root), feeder=feeder)

    # Sweep + headline aggregate
    thrs = np.linspace(0.0, 1.0, 11)
    frames = [sweep(m.parent, thrs) for m in runs_root.rglob("manifest.json")]
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_root / "physical_metrics.csv", index=False)
        head = df[df["threshold"].between(0.49, 0.51)]
        agg = (head.groupby(["scenario", "detector"], as_index=False)
                  [["ens_kwh", "curt_kwh", "voltage_frac", "freq_dev_hz"]]
                  .mean())
        agg.to_csv(out_root / "baseline_comparison.csv", index=False)

    # Detection + Caution
    det_frames = [detection_aggregate(m.parent) for m in runs_root.rglob("manifest.json")]
    if det_frames:
        det_df = pd.concat(det_frames, ignore_index=True)
        det_df.to_csv(out_root / "detection_metrics.csv", index=False)
        rows = []
        for (scn, detn, sd), g in det_df.groupby(["scenario", "detector", "seed"]):
            rows.append({"scenario": scn, "detector": detn, "seed": sd,
                          "macro_f1": macro_f1(g)})
        pd.DataFrame(rows).to_csv(out_root / "macro_f1_by_run.csv", index=False)

    cau = [caution_aggregate(m.parent) for m in runs_root.rglob("manifest.json")]
    if cau:
        pd.DataFrame(cau).to_csv(out_root / "caution_metrics.csv", index=False)

    readme = ["# IJCIP baselines comparison\n",
                "Detectors compared on the same 3-scenario × 5-seed matrix as",
                "the headline experiments. Oracle is an upper-bound reference,",
                "not a deployable system.\n",
                "Detectors:"]
    for d in DETECTORS:
        readme.append(f"  - {d}")
    (out_root / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote baseline comparison CSV with {len(DETECTORS)} detectors")


if __name__ == "__main__":
    main()
