"""Run the full experimental matrix.

For each (scenario, detector, seed):
  1. Run the harness.
  2. Sweep detection threshold against the four physical-impact metrics
     (ENS, curtailment, voltage-band excursion, frequency deviation).
  3. Compute per-class detection metrics.
  4. Compute Caution-Agent FP/FN/unsafe rates.
  5. Compute paired-bootstrap CIs on each physical metric vs each
     baseline (Holm-corrected p-values).
  6. Aggregate runtime statistics.

No monetary aggregation is performed; every row carries its physical
metric in native units (kWh, fraction, Hz).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .caution_metrics import aggregate_run as caution_aggregate
from .detection_metrics import aggregate_run as detection_aggregate, macro_f1
from .physical_curves import sweep
from .stats.aggregate import aggregate_seeds
from .stats.bootstrap import paired_bootstrap_ci
from .stats.multitest import holm_correct
from ..simulation.feeder import StubFeeder
from ..simulation.harness import run_scenario
from ..simulation.hitl import OperatorApprover


DETECTORS = {
    "rule_ids":     "code.baselines.rule_ids.adapter:RuleIDS",
    "single_llm":   "code.baselines.single_llm.runner:SingleLLM",
    "prior_mas":    "code.baselines.prior_mas.adapter:PriorMAS",
    "der_secagent": "code.Multi_AI_Agent.adapter:DERSecAgentDetector",
}

SCENARIOS = [
    "code/simulation/scenarios/ieee13_fdi_inverter/config.yaml",
    "code/simulation/scenarios/ieee13_command_spoof/config.yaml",
    "code/simulation/scenarios/ieee34_command_spoof_derms/config.yaml",
]

# Headline physical metric for the bootstrap analysis.
HEADLINE_METRIC = "ens_kwh"


def _load_detector(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--use-stub", action="store_true",
                    help="run with StubFeeder (no OpenDSS); for CI smoke-runs")
    ap.add_argument("--out", default="code/results")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = Path(args.out)
    out_root.mkdir(exist_ok=True)
    runs_root = out_root / "runs"

    # 1. run harness for every (scenario, detector, seed).
    for scenario in SCENARIOS:
        for det_name, det_spec in DETECTORS.items():
            for seed in seeds:
                feeder = None
                if args.use_stub:
                    import yaml
                    cfg = yaml.safe_load(Path(scenario).read_text())
                    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"])
                detector = _load_detector(det_spec)
                operator = OperatorApprover() if det_name == "der_secagent" else None
                run_scenario(scenario, detector, seed, out_root=str(runs_root),
                             feeder=feeder, operator=operator)

    # 2. physical-metric threshold sweep
    thrs = np.linspace(0.0, 1.0, 11)
    frames = [sweep(m.parent, thrs) for m in runs_root.rglob("manifest.json")]
    physical_df = pd.concat(frames, ignore_index=True)
    physical_df.to_csv(out_root / "physical_metrics.csv", index=False)

    # 3. headline aggregates at threshold=0.5 --- one CSV per metric.
    head = physical_df[physical_df["threshold"].between(0.49, 0.51)]
    for metric in ("ens_kwh", "curt_kwh", "voltage_frac", "freq_dev_hz"):
        agg = aggregate_seeds(head, metric)
        agg.to_csv(out_root / f"headline_{metric}.csv", index=False)

    # 4. per-class detection metrics
    det_frames = [detection_aggregate(m.parent) for m in runs_root.rglob("manifest.json")]
    if det_frames:
        det_df = pd.concat(det_frames, ignore_index=True)
        det_df.to_csv(out_root / "detection_metrics.csv", index=False)
        rows = []
        for (scn, detn, sd), g in det_df.groupby(["scenario", "detector", "seed"]):
            rows.append({"scenario": scn, "detector": detn, "seed": sd,
                         "macro_f1": macro_f1(g)})
        pd.DataFrame(rows).to_csv(out_root / "macro_f1_by_run.csv", index=False)

    # 5. Caution-Agent quantitative metrics
    cau_rows = [caution_aggregate(m.parent) for m in runs_root.rglob("manifest.json")]
    if cau_rows:
        pd.DataFrame(cau_rows).to_csv(out_root / "caution_metrics.csv", index=False)

    # 6. pairwise bootstrap on the headline physical metric.
    if "der_secagent" in head["detector"].unique() and len(head["detector"].unique()) >= 2:
        ref = "der_secagent"
        rows = []
        for det in head["detector"].unique():
            if det == ref:
                continue
            rows.append(paired_bootstrap_ci(head, HEADLINE_METRIC, det, ref))
        if rows:
            corrected = holm_correct([r["p_one_sided"] for r in rows])
            for r, c in zip(rows, corrected):
                r["p_adj_holm"] = c["p_adj"]
                r["reject"] = c["reject"]
        with open(out_root / "pairwise_bootstrap.json", "w") as f:
            json.dump(rows, f, indent=2)

    # 7. runtime aggregate
    runtime_rows = []
    for m in runs_root.rglob("manifest.json"):
        rt = m.parent / "runtime.json"
        if not rt.exists():
            continue
        manifest = json.loads(m.read_text())
        rec = json.loads(rt.read_text())
        runtime_rows.append({"scenario": manifest["scenario"],
                             "detector": manifest["detector"],
                             "seed": manifest["seed"], **rec})
    if runtime_rows:
        pd.DataFrame(runtime_rows).to_csv(out_root / "runtime.csv", index=False)

    print("done")


if __name__ == "__main__":
    main()
