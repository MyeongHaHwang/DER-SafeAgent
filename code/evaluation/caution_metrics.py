"""Caution-Agent quantitative metrics — answers paper §5.4.

Three rates per (scenario × detector × seed):
  - fp_action_rate: fraction of issued actions during steps with no active
    attack (executing a mitigation against nothing).
  - fn_action_rate: fraction of attack steps with no issued action
    (failing to act when the attack was real).
  - unsafe_action_rate: fraction of issued actions whose post-hoc
    expected_impact_tier (carried in the action's params) is "severe".

These are descriptive — they isolate the safety-layer contribution of the
Caution Agent from raw classification accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def caution_rates(decisions: list[dict]) -> dict[str, float]:
    n_attack_steps = n_quiet_steps = 0
    n_action_during_attack = n_action_during_quiet = 0
    n_actions_total = n_unsafe = 0

    for d in decisions:
        is_attack = bool(d.get("ground_truth_active_attacks"))
        actions = d.get("actions", [])
        has_action = any(a.get("name", "no_op") != "no_op" for a in actions)

        if is_attack:
            n_attack_steps += 1
            if has_action:
                n_action_during_attack += 1
        else:
            n_quiet_steps += 1
            if has_action:
                n_action_during_quiet += 1

        for a in actions:
            if a.get("name", "no_op") == "no_op":
                continue
            n_actions_total += 1
            tier = a.get("params", {}).get("impact_tier")
            if tier == "severe":
                n_unsafe += 1

    fp_rate = n_action_during_quiet / n_quiet_steps if n_quiet_steps else 0.0
    fn_rate = 1.0 - (n_action_during_attack / n_attack_steps) if n_attack_steps else 0.0
    unsafe_rate = n_unsafe / n_actions_total if n_actions_total else 0.0
    return {
        "fp_action_rate": fp_rate,
        "fn_action_rate": fn_rate,
        "unsafe_action_rate": unsafe_rate,
        "n_actions_total": float(n_actions_total),
        "n_attack_steps": float(n_attack_steps),
        "n_quiet_steps": float(n_quiet_steps),
    }


def aggregate_run(run_dir: Path) -> dict:
    decisions = [json.loads(line) for line in
                 (run_dir / "decisions.jsonl").read_text().splitlines() if line]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rates = caution_rates(decisions)
    rates.update({"scenario": manifest["scenario"],
                  "detector": manifest["detector"],
                  "seed": manifest["seed"]})
    return rates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="code/results/runs")
    ap.add_argument("--out", default="code/results/caution_metrics.csv")
    args = ap.parse_args()

    rows = []
    for manifest in Path(args.runs_root).rglob("manifest.json"):
        rows.append(aggregate_run(manifest.parent))
    if not rows:
        print("no runs found")
        return
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
