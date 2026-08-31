"""Real-LoRA latency benchmark with one-process-per-backend isolation.

The shipped ``run_llm_latency_p0.py`` loads every backend into the
same Python process, which OOMs a 10 GB consumer GPU. This wrapper
spawns a subprocess per (backend, K) cell so each model is loaded
on a clean device, then merges every per-cell CSV into the same
``latency_metrics.csv`` file the figure / table generators consume.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


def _quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(round(q * (len(s) - 1)))
    return s[k]


def _bench(call: Callable[[], dict], n: int) -> dict:
    latencies = []
    schema_fail = 0
    json_fail = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            out = call()
            if not isinstance(out, dict):
                json_fail += 1
            if not (isinstance(out, dict) and "attack_class" in out):
                schema_fail += 1
        except Exception:
            json_fail += 1
            schema_fail += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total = sum(latencies)
    return {
        "n_calls": n,
        "mean_ms": statistics.mean(latencies) if latencies else float("nan"),
        "p50_ms": _quantile(latencies, 0.50),
        "p95_ms": _quantile(latencies, 0.95),
        "p99_ms": _quantile(latencies, 0.99),
        "max_ms": max(latencies) if latencies else float("nan"),
        "throughput_hz": (1000.0 * n / total) if total > 0 else 0.0,
        "schema_failure_rate": schema_fail / n if n else float("nan"),
        "json_valid_rate": 1.0 - (json_fail / n) if n else float("nan"),
    }


def _gpu_mem():
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    return float("nan")


# --- subprocess entry point ----------------------------------------------

REPRESENTATIVE_PROMPTS = [
    '{"feature_view":{"dominant_signal":"tampered"},"event_window":[{"kind":"telemetry","tampered":true,"payload":{"asset":"INV_634","p_kw":200}}],"memory":"(none)"}',
    '{"feature_view":{"dominant_signal":"command"},"event_window":[{"kind":"command","tampered":true,"payload":{"asset":"INV_634","type":"setpoint","p_kw":25,"from":"DERMS"}}],"memory":"(none)"}',
    '{"feature_view":{"dominant_signal":"freeze","persistent_freeze":true},"event_window":[],"memory":"(none)"}',
]


def _child_run(backend: str, k: int, n: int) -> dict:
    """Run inside a fresh process so GPU memory is reclaimed on exit."""
    counter = {"i": 0}
    if backend == "heuristic_fallback":
        from code.llm_serving.local_lora import _heuristic_decision
        def call():
            out = _heuristic_decision(REPRESENTATIVE_PROMPTS[counter["i"] % 3])
            counter["i"] += 1
            return out
    else:
        env_base, env_adapter = {
            "qwen2_5_7b_base":   ("DER_BASE_QWEN7B",  None),
            "qwen2_5_7b_lora":   ("DER_BASE_QWEN7B",  "DER_LORA_ADAPTER_QWEN7B"),
            "llama3_1_8b_base":  ("DER_BASE_LLAMA8B", None),
            "llama3_1_8b_lora":  ("DER_BASE_LLAMA8B", "DER_LORA_ADAPTER_LLAMA8B"),
        }[backend]
        base = os.environ.get(env_base, "")
        adapter = os.environ.get(env_adapter, "") if env_adapter else ""
        if not base:
            return {"backend": backend, "k": f"k={k}", "backend_status": "not_run",
                    "not_run_reason": f"env {env_base} unset",
                    "n_calls": 0,
                    "mean_ms": float("nan"), "p50_ms": float("nan"),
                    "p95_ms": float("nan"), "p99_ms": float("nan"),
                    "max_ms": float("nan"), "throughput_hz": float("nan"),
                    "schema_failure_rate": float("nan"),
                    "json_valid_rate": float("nan"),
                    "gpu_mem_alloc_mb": float("nan"),
                    "cpu_mem_rss_mb": float("nan")}
        from code.llm_serving.local_lora import LocalLoRA
        backend_obj = LocalLoRA(base_model=base, adapter_path=(adapter or None))
        loaded = backend_obj._try_load()
        if not loaded:
            return {"backend": backend, "k": f"k={k}", "backend_status": "failed",
                    "not_run_reason": "model failed to load",
                    "n_calls": 0,
                    "mean_ms": float("nan"), "p50_ms": float("nan"),
                    "p95_ms": float("nan"), "p99_ms": float("nan"),
                    "max_ms": float("nan"), "throughput_hz": float("nan"),
                    "schema_failure_rate": float("nan"),
                    "json_valid_rate": float("nan"),
                    "gpu_mem_alloc_mb": float("nan"),
                    "cpu_mem_rss_mb": float("nan")}
        def call():
            out = backend_obj.generate_json(REPRESENTATIVE_PROMPTS[counter["i"] % 3])
            counter["i"] += 1
            return out
        # warmup once so JIT compile latency doesn't dominate the small sample
        call()
    stats = _bench(lambda: [call() for _ in range(k)][-1], n)
    stats.update({"backend": backend, "k": f"k={k}",
                  "backend_status": "ok", "not_run_reason": "",
                  "gpu_mem_alloc_mb": _gpu_mem(),
                  "cpu_mem_rss_mb": float("nan")})
    return stats


# --- parent driver -------------------------------------------------------

BACKEND_ORDER = ["heuristic_fallback",
                  "qwen2_5_7b_base", "qwen2_5_7b_lora",
                  "llama3_1_8b_base", "llama3_1_8b_lora"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=30)
    ap.add_argument("--heuristic-n-calls", type=int, default=200)
    ap.add_argument("--out", default="code/results/ijcip_latency")
    ap.add_argument("--child", default=None,
                     help="internal: run a single (backend, K) cell")
    ap.add_argument("--child-k", type=int, default=1)
    ap.add_argument("--child-n", type=int, default=30)
    args = ap.parse_args()

    if args.child:
        row = _child_run(args.child, args.child_k, args.child_n)
        sys.stdout.write("__RESULT__" + json.dumps(row) + "\n")
        return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in BACKEND_ORDER:
        for k in (1, 3):
            n = args.heuristic_n_calls if backend == "heuristic_fallback" else args.n_calls
            cmd = [sys.executable, "-m",
                    "code.evaluation.latency.run_llm_latency_real",
                    "--child", backend, "--child-k", str(k),
                    "--child-n", str(n)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            row = None
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT__"):
                    row = json.loads(line[len("__RESULT__"):])
                    break
            if row is None:
                row = {"backend": backend, "k": f"k={k}",
                        "backend_status": "failed",
                        "not_run_reason": f"subprocess error: {proc.stderr[-200:]}",
                        "n_calls": 0,
                        "mean_ms": float("nan"), "p50_ms": float("nan"),
                        "p95_ms": float("nan"), "p99_ms": float("nan"),
                        "max_ms": float("nan"), "throughput_hz": float("nan"),
                        "schema_failure_rate": float("nan"),
                        "json_valid_rate": float("nan"),
                        "gpu_mem_alloc_mb": float("nan"),
                        "cpu_mem_rss_mb": float("nan")}
            rows.append(row)
            print(f"{backend} k={k}: status={row['backend_status']} "
                  f"mean_ms={row.get('mean_ms')}")
    field_order = ["n_calls", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                    "max_ms", "throughput_hz", "schema_failure_rate",
                    "json_valid_rate", "gpu_mem_alloc_mb", "cpu_mem_rss_mb",
                    "backend", "k", "backend_status", "not_run_reason"]
    with (out / "latency_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in field_order})
    print(f"wrote {out / 'latency_metrics.csv'}")


if __name__ == "__main__":
    main()
