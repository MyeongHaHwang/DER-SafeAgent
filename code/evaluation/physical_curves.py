"""Threshold sweep over raw physical-impact metrics.

Replaces the earlier USD-aggregated module so that no monetary weighting
appears anywhere in the evaluation pipeline. We report the four physical
metrics directly (ENS, curtailment, voltage band excursion, frequency
deviation) and detection-confusion counts at each threshold.

Inputs (per (scenario, detector, seed) run):
- timeseries.csv  --- for energy_metrics
- decisions.jsonl --- for detection events and ground truth

Output: one row per (scenario, detector, seed, threshold) --- long-format CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .energy_metrics import (
    curtailed_energy,
    energy_not_supplied,
    frequency_excursion,
    ramp_violations,
    voltage_deviation,
)


def physical_summary(ts: pd.DataFrame) -> dict[str, float]:
    return {
        "ens_kwh":           energy_not_supplied(ts).value,
        "curt_kwh":          curtailed_energy(ts).value,
        "voltage_frac":      voltage_deviation(ts).value,
        "freq_dev_hz":       frequency_excursion(ts).value,
        "ramp_violations":   ramp_violations(ts).value,
    }


def _confusion(decisions: list[dict], threshold: float) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for d in decisions:
        gt = bool(d.get("ground_truth_active_attacks"))
        conf = float(d["detection"].get("confidence", 0.0))
        pred = (d["detection"].get("attack_class", "none") != "none") and (conf >= threshold)
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif (not gt) and pred:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def sweep(run_dir: Path, thresholds: np.ndarray) -> pd.DataFrame:
    ts = pd.read_csv(run_dir / "timeseries.csv")
    decisions = [json.loads(line) for line in
                 (run_dir / "decisions.jsonl").read_text().splitlines() if line]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = physical_summary(ts)

    rows = []
    for thr in thresholds:
        cm = _confusion(decisions, float(thr))
        rows.append({
            "scenario": manifest["scenario"],
            "detector": manifest["detector"],
            "seed": manifest["seed"],
            "threshold": float(thr),
            **cm, **summary,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="code/results/runs")
    ap.add_argument("--out", default="code/results/physical_metrics.csv")
    ap.add_argument("--thresholds",
                    default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = ap.parse_args()

    thrs = np.array([float(x) for x in args.thresholds.split(",")])
    runs_root = Path(args.runs_root)
    frames = []
    for manifest in runs_root.rglob("manifest.json"):
        frames.append(sweep(manifest.parent, thrs))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
