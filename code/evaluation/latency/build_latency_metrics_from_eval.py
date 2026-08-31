"""Compose the IJCIP latency_metrics.csv from honest sources.

Heuristic-fallback rows are measured directly here on representative
prompts. Real-LoRA rows are sourced from the per-adapter
``eval_lora`` artefacts under ``code/results/ijcip_lora_eval``,
which already measured K=1 and K=3 end-to-end on the test split
with the chat-template applied.

This avoids the ``apply_chat_template``-vs-raw-prompt mismatch that
made the standalone latency runner generate full ``max_new_tokens``
completions on tuned chat models.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path


REPRESENTATIVE_PROMPTS = [
    '{"feature_view":{"dominant_signal":"tampered"},"event_window":[{"kind":"telemetry","tampered":true,"payload":{"asset":"INV_634","p_kw":200}}],"memory":"(none)"}',
    '{"feature_view":{"dominant_signal":"command"},"event_window":[{"kind":"command","tampered":true,"payload":{"asset":"INV_634","type":"setpoint","p_kw":25,"from":"DERMS"}}],"memory":"(none)"}',
    '{"feature_view":{"dominant_signal":"freeze","persistent_freeze":true},"event_window":[],"memory":"(none)"}',
]


def _q(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[int(round(q * (len(s) - 1)))]


def _heuristic_row(k: int, n: int) -> dict:
    from code.llm_serving.local_lora import _heuristic_decision
    latencies = []
    json_fail = 0
    schema_fail = 0
    for i in range(n):
        t0 = time.perf_counter()
        out = None
        try:
            for _ in range(k):
                out = _heuristic_decision(REPRESENTATIVE_PROMPTS[i % 3])
        except Exception:
            json_fail += 1
            schema_fail += 1
        if not isinstance(out, dict):
            json_fail += 1
        elif "attack_class" not in out:
            schema_fail += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total = sum(latencies)
    return {
        "n_calls": n,
        "mean_ms": statistics.mean(latencies) if latencies else float("nan"),
        "p50_ms": _q(latencies, 0.50),
        "p95_ms": _q(latencies, 0.95),
        "p99_ms": _q(latencies, 0.99),
        "max_ms": max(latencies) if latencies else float("nan"),
        "throughput_hz": (1000.0 * n / total) if total > 0 else 0.0,
        "schema_failure_rate": schema_fail / n if n else float("nan"),
        "json_valid_rate": 1.0 - (json_fail / n) if n else float("nan"),
        "gpu_mem_alloc_mb": 0.0,
        "cpu_mem_rss_mb": float("nan"),
        "backend": "heuristic_fallback",
        "k": f"k={k}",
        "backend_status": "ok",
        "not_run_reason": "",
    }


def _lora_row(tag: str, backend_name: str, k: int) -> dict:
    p = Path("code/results/ijcip_lora_eval") / tag / "eval_metrics.json"
    if not p.exists():
        return {
            "n_calls": 0,
            "mean_ms": float("nan"), "p50_ms": float("nan"),
            "p95_ms": float("nan"), "p99_ms": float("nan"),
            "max_ms": float("nan"), "throughput_hz": float("nan"),
            "schema_failure_rate": float("nan"),
            "json_valid_rate": float("nan"),
            "gpu_mem_alloc_mb": float("nan"),
            "cpu_mem_rss_mb": float("nan"),
            "backend": backend_name, "k": f"k={k}",
            "backend_status": "not_run",
            "not_run_reason": f"missing {p}",
        }
    m = json.loads(p.read_text())
    if k == 1:
        mean = m["latency_k1_ms_mean"]; p95 = m["latency_k1_ms_p95"]
    else:
        mean = m["latency_kK_ms_mean"]; p95 = m["latency_kK_ms_p95"]
    return {
        "n_calls": m["n_samples"],
        "mean_ms": mean,
        "p50_ms": float("nan"),
        "p95_ms": p95,
        "p99_ms": float("nan"),
        "max_ms": float("nan"),
        "throughput_hz": 1000.0 / mean if mean else float("nan"),
        "schema_failure_rate": 1.0 - m["schema_compliance_rate"],
        "json_valid_rate": m["json_validity_rate"],
        "gpu_mem_alloc_mb": float("nan"),
        "cpu_mem_rss_mb": float("nan"),
        "backend": backend_name, "k": f"k={k}",
        "backend_status": "ok", "not_run_reason": "",
    }


def main() -> None:
    out_dir = Path("code/results/ijcip_latency"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # heuristic fallback rows are measured here, fast.
    for k in (1, 3):
        rows.append(_heuristic_row(k, n=200))
    # LoRA rows are sourced from the eval_lora artefacts.
    for tag, name in (("qwen2_5_7b_lora", "qwen2_5_7b_lora"),
                        ("llama3_1_8b_lora", "llama3_1_8b_lora")):
        for k in (1, 3):
            rows.append(_lora_row(tag, name, k))

    field_order = ["n_calls", "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                    "max_ms", "throughput_hz", "schema_failure_rate",
                    "json_valid_rate", "gpu_mem_alloc_mb", "cpu_mem_rss_mb",
                    "backend", "k", "backend_status", "not_run_reason"]
    with (out_dir / "latency_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in field_order})
    print(f"wrote {out_dir / 'latency_metrics.csv'} with {len(rows)} rows")
    for r in rows:
        print(f"  {r['backend']} {r['k']}: status={r['backend_status']} "
                f"mean_ms={r['mean_ms']}")


if __name__ == "__main__":
    main()
