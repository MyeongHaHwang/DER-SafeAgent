"""IJCIP P0 expanded adversarial-robustness driver.

* 14 perturbation families (see ``perturbations_expanded.py``)
* 8 detectors (rule_ids, single_llm, prior_mas, safe_single_llm,
  deterministic_energy_policy, single_llm_with_caution,
  prior_mas_with_safety, der_secagent)
* configurable cases-per-family and seeds

Outputs (under ``code/results/ijcip_adversarial_safety/``):
    robustness_metrics_expanded.csv
    family_breakdown_expanded.csv
    seed_breakdown.csv
    failure_examples.jsonl
    README.md  (extended)
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import pandas as pd

from .perturbations_expanded import ALL_GENERATORS, build_expanded_suite
from .run_robustness import (DETECTOR_FACTORIES, _evaluate_violation,
                              _instantiate, _step_detector)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors",
                    default=",".join(DETECTOR_FACTORIES.keys()))
    ap.add_argument("--n-per-family", type=int, default=50)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="code/results/ijcip_adversarial_safety")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    failure_log = out / "failure_examples.jsonl"
    failure_log.write_text("")

    metric_rows = []         # (detector, family) aggregated over seeds
    by_seed_rows = []        # (detector, family, seed)

    for seed in seeds:
        cases = build_expanded_suite(seed=seed, n_per_family=args.n_per_family)
        for det_name in [d for d in args.detectors.split(",") if d.strip()]:
            spec = DETECTOR_FACTORIES.get(det_name)
            if spec is None:
                continue
            for fam in ALL_GENERATORS.keys():
                sub = [c for c in cases if c.perturbation == fam]
                detector = _instantiate(spec)
                agg = {"policy_violation": 0, "forbidden_action": 0,
                        "unauthorized_command": 0, "schema_failure": 0,
                        "safe_fallback": 0, "hitl_escalation": 0,
                        "abstained": 0, "correct_refusal": 0,
                        "recovery_success": 0,
                        "attack_success": 0}
                for c in sub:
                    rec = _step_detector(detector, c)
                    flags = _evaluate_violation(c, rec["action"], rec["tier"],
                                                  rec["hitl"], rec["schema_ok"])
                    flags["attack_success"] = 1 if flags["policy_violation"] else 0
                    for k in agg:
                        agg[k] += flags.get(k, 0)
                    if flags["policy_violation"]:
                        with failure_log.open("a") as fh:
                            fh.write(json.dumps({"seed": seed,
                                                    "detector": det_name,
                                                    "family": fam,
                                                    **rec, **flags}) + "\n")
                n = max(len(sub), 1)
                row = {
                    "seed":         seed,
                    "detector":     det_name,
                    "perturbation": fam,
                    "n_cases":      len(sub),
                    **{f"{k}_rate": agg[k] / n for k in agg},
                }
                by_seed_rows.append(row)

    by_seed_df = pd.DataFrame(by_seed_rows)
    by_seed_df.to_csv(out / "seed_breakdown.csv", index=False)

    rate_cols = [c for c in by_seed_df.columns if c.endswith("_rate")]
    fam_df = (by_seed_df.groupby(["detector", "perturbation"], as_index=False)
                          [rate_cols].mean())
    fam_df.to_csv(out / "family_breakdown_expanded.csv", index=False)

    overall = (by_seed_df.groupby(["detector"], as_index=False)
                          [rate_cols].mean())
    overall.to_csv(out / "robustness_metrics_expanded.csv", index=False)

    readme = ["# IJCIP P0 expanded adversarial robustness\n",
                f"- families: {len(ALL_GENERATORS)}",
                f"- cases per family: {args.n_per_family}",
                f"- seeds: {seeds}",
                f"- detectors: {args.detectors}",
                "",
                "Files:",
                "  robustness_metrics_expanded.csv  --- per-detector aggregate",
                "  family_breakdown_expanded.csv    --- per-(detector, family) aggregate",
                "  seed_breakdown.csv               --- per-(detector, family, seed) raw rows",
                "  failure_examples.jsonl           --- one line per policy violation"]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote expanded adversarial: {len(by_seed_df)} seed-rows, "
            f"{len(fam_df)} family-rows, {len(overall)} detector rows")


if __name__ == "__main__":
    main()
