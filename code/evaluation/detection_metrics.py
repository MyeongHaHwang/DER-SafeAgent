"""Per-class detection metrics (precision / recall / F1) over the attack taxonomy.

Consumes the per-step ``decisions.jsonl`` emitted by the harness. Ground truth
is the union of attack-injector activity for that step (any non-empty
``ground_truth_active_attacks`` is positive). Per-class metrics use
one-vs-rest matching of the predicted ``attack_class`` against the true
class indicated by the active injector(s); steps where no injector is
active and the prediction is also ``none`` are true-negatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLASSES = ["none", "fdi", "replay", "command_spoof", "dos", "firmware"]


def _ground_truth_class(record: dict) -> str:
    active = record.get("ground_truth_active_attacks", [])
    if not active:
        return "none"
    # Injector names map directly to attack classes in this harness.
    return active[0]


def per_class_metrics(decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for cls in CLASSES:
        tp = fp = fn = 0
        for d in decisions:
            true_cls = _ground_truth_class(d)
            pred_cls = d["detection"].get("attack_class", "none")
            if pred_cls == cls and true_cls == cls:
                tp += 1
            elif pred_cls == cls and true_cls != cls:
                fp += 1
            elif pred_cls != cls and true_cls == cls:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append({"class": cls, "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)


def macro_f1(per_class: pd.DataFrame) -> float:
    """Macro-F1 over non-trivial classes (skip 'none' which dominates)."""
    sub = per_class[per_class["class"] != "none"]
    return float(sub["f1"].mean()) if len(sub) else 0.0


def aggregate_run(run_dir: Path) -> pd.DataFrame:
    decisions = [json.loads(line) for line in
                 (run_dir / "decisions.jsonl").read_text().splitlines() if line]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    df = per_class_metrics(decisions)
    df["scenario"] = manifest["scenario"]
    df["detector"] = manifest["detector"]
    df["seed"] = manifest["seed"]
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="code/results/runs")
    ap.add_argument("--out", default="code/results/detection_metrics.csv")
    args = ap.parse_args()

    frames = []
    for manifest in Path(args.runs_root).rglob("manifest.json"):
        frames.append(aggregate_run(manifest.parent))
    if not frames:
        print("no runs found")
        return
    df = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
