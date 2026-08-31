#!/usr/bin/env python3
"""Re-derive the canonical consultation-latency distribution (Table 11).

The manuscript's latency table is built from
``code/results/ijcip_final/latency/latency_distributions.csv``.  That file is
a summary of the per-call latencies recorded (as ``latency_ms``) in the four
archived strict-serving evaluation runs under
``code/results/ijcip_revision_r1r2_20260805/lora_eval/``:

    qwen2_5_7b_lora_k1 / _k3_subset48, llama3_1_8b_lora_k1 / _k3_subset48

This script recomputes the summary from those raw predictions and, by
default, only *verifies* that the recomputation matches the shipped canonical
CSV value-for-value (it does not overwrite anything).  Use ``--write`` to
emit the recomputed file to ``artifacts/processed/latency_distributions.csv``.

Verified: the recomputation is value-identical to the shipped canonical file
(same rounding, same row order).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LORA_EVAL = ROOT / "code/results/ijcip_revision_r1r2_20260805/lora_eval"
CANONICAL = ROOT / "code/results/ijcip_final/latency/latency_distributions.csv"
OUT = ROOT / "artifacts/processed/latency_distributions.csv"

RUNS = [("qwen", 1, "qwen2_5_7b_lora_k1"),
        ("qwen", 3, "qwen2_5_7b_lora_k3_subset48"),
        ("llama", 1, "llama3_1_8b_lora_k1"),
        ("llama", 3, "llama3_1_8b_lora_k3_subset48")]


def derive() -> pd.DataFrame:
    rows = []
    for backend, k, run in RUNS:
        preds = LORA_EVAL / run / "predictions.jsonl"
        lat = np.array([json.loads(line)["latency_ms"] / 1000.0
                        for line in preds.open()])
        rows.append({
            "backend": backend, "k": k, "n_calls": len(lat),
            "mean_s": round(lat.mean(), 3),
            "median_s": round(float(np.median(lat)), 3),
            "sd_s": round(float(lat.std(ddof=1)), 3),
            "p90_s": round(float(np.percentile(lat, 90)), 3),
            "p95_s": round(float(np.percentile(lat, 95)), 3),
            "p99_s": round(float(np.percentile(lat, 99)), 3),
            "max_s": round(float(lat.max()), 3),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help=f"also write the recomputed CSV to {OUT}")
    args = ap.parse_args()

    got = derive()
    want = pd.read_csv(CANONICAL)
    if not got.reset_index(drop=True).equals(want.reset_index(drop=True)):
        print("[FAIL] recomputed latency summary differs from the canonical "
              "file:", file=sys.stderr)
        print("--- recomputed ---\n" + got.to_csv(index=False), file=sys.stderr)
        print("--- canonical ---\n" + want.to_csv(index=False), file=sys.stderr)
        return 1
    print("[ok] latency summary re-derived value-identically from the four "
          "archived predictions.jsonl runs")
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        got.to_csv(OUT, index=False)
        print(f"[csv] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
