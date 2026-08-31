"""V3 Phase 5 — deadline-aware latency rebuild from measured per-call data.

Reconciles the v2 "~8 s" text and "13.7 s" table by reporting per-backbone,
per-K latency DISTRIBUTIONS from the canonical 146-call measurements
(lora_eval predictions.jsonl), with adapter-load time reported separately from
per-call inference. SLO admission {1,5,10,15,30,60}s under fast-path-first
semantics: the deterministic fast path acts at t=0; a model result is adopted
only if its measured latency < SLO (and would pass safety); an expired result
never rewrites history.

Run: python3 -m code.evaluation.final_safeagent_v3.run_deadline_v3
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LORA = Path("code/results/ijcip_revision_r1r2_20260805/lora_eval")
OUT = Path("code/results/ijcip_final_v3/deadline")
SLOS = [1, 5, 10, 15, 30, 60]


def _latencies(subdir):
    p = LORA / subdir / "predictions.jsonl"
    if not p.exists():
        return None, None
    lat = [json.loads(l).get("latency_ms") for l in p.read_text().splitlines()]
    lat = np.array([x for x in lat if x is not None], float) / 1000.0
    man = LORA / subdir / "run_manifest.json"
    load_s = None
    if man.exists():
        m = json.loads(man.read_text())
        load_s = m.get("adapter_load_s") or m.get("model_load_s")
    return lat, load_s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subdirs = {
        ("qwen", 1): "qwen2_5_7b_lora_k1",
        ("qwen", 3): "qwen2_5_7b_lora_k3_subset48",
        ("llama", 1): "llama3_1_8b_lora_k1",
        ("llama", 3): "llama3_1_8b_lora_k3_subset48",
    }
    dist_rows, slo_rows = [], []
    for (be, k), sub in subdirs.items():
        lat, load_s = _latencies(sub)
        if lat is None:
            continue
        dist_rows.append({
            "backend": be, "k": k, "n_calls": len(lat),
            "mean_s": float(lat.mean()), "median_s": float(np.median(lat)),
            "sd_s": float(lat.std(ddof=1)), "p90_s": float(np.quantile(lat, 0.90)),
            "p95_s": float(np.quantile(lat, 0.95)), "max_s": float(lat.max()),
            "adapter_load_s": load_s})
        for slo in SLOS:
            admit = float((lat < slo).mean())
            slo_rows.append({"backend": be, "k": k, "slo_s": slo,
                             "admit_rate": admit,
                             "timeout_rate": 1.0 - admit})
    dd = pd.DataFrame(dist_rows)
    ss = pd.DataFrame(slo_rows)
    dd.round(3).to_csv(OUT / "latency_distributions.csv", index=False)
    ss.round(3).to_csv(OUT / "slo_admission.csv", index=False)
    note = {
        "reconciliation": ("v2 text '~8 s' and Table 10 '13.7 s' are both "
                           "superseded; per-call inference is reported per "
                           "backbone/K below. Adapter-load time is reported "
                           "separately and is NOT part of per-call latency."),
        "k1_mean_s": {r["backend"]: r["mean_s"]
                      for r in dist_rows if r["k"] == 1},
        "k3_mean_s": {r["backend"]: r["mean_s"]
                      for r in dist_rows if r["k"] == 3},
        "semantics": ("fast path acts at t=0; model result adopted only if "
                      "measured latency < SLO and it passes safety; expired "
                      "result never rewrites history"),
    }
    (OUT / "latency_note.json").write_text(json.dumps(note, indent=2))
    print(dd.round(2).to_string(index=False))
    print()
    print(ss.pivot_table(index=["backend", "k"], columns="slo_s",
                         values="admit_rate").round(2).to_string())
    print(json.dumps(note["k1_mean_s"], indent=2))


if __name__ == "__main__":
    main()
