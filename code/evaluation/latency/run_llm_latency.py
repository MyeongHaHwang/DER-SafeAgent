"""Measure end-to-end LLM latency for the Hypothesis Agent under three
backends:

    heuristic_fallback         --- the bundled deterministic stub.
    qwen2_5_7b_lora            --- LocalLoRA(Qwen2.5-7B-Instruct + adapter).
    llama3_1_8b_lora           --- LocalLoRA(Llama-3.1-8B-Instruct + adapter).

For each available backend we run K=1 (single-strategy) and K=3 (the
production self-consistency aggregator). When a backbone or its LoRA
adapter is not available, the corresponding row is recorded with
status="not_run" and a reason --- *no* numbers are fabricated.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import pandas as pd


def _benchmark(call: Callable[[], dict], n: int) -> dict:
    latencies = []
    schema_failures = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            out = call()
            ok = isinstance(out, dict) and "attack_class" in out
            schema_failures += 0 if ok else 1
        except Exception:
            schema_failures += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        "n_calls":        n,
        "mean_ms":        statistics.mean(latencies),
        "p50_ms":         statistics.median(latencies),
        "p95_ms":         _quantile(latencies, 0.95),
        "p99_ms":         _quantile(latencies, 0.99),
        "max_ms":         max(latencies),
        "throughput_hz":  1000.0 * n / sum(latencies) if sum(latencies) > 0 else 0.0,
        "schema_failure_rate": schema_failures / n,
    }


def _quantile(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = int(q * (len(s) - 1))
    return s[k]


def _build_call_heuristic():
    from ...llm_serving.local_lora import get_default
    backend = get_default()
    prompt = json.dumps({
        "feature_view": {"dominant_signal": "tampered",
                          "n_tampered_events": 3, "severity_score": 0.8},
        "telemetry_window": [], "event_window": [
            {"kind": "telemetry", "tampered": True,
              "payload": {"asset": "INV_634", "p_kw": 200}}
        ],
        "memory": "(none)",
    })
    return lambda: backend.generate_json(prompt)


def _build_call_real_lora(base_model: str, adapter_env: str):
    """Try to load the production adapter; return None if unavailable."""
    adapter_path = os.environ.get(adapter_env)
    if not adapter_path or not Path(adapter_path).exists():
        return None, f"env {adapter_env} unset or path missing"
    try:
        import torch                               # noqa: F401
        from peft import PeftModel                 # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    except ImportError as exc:
        return None, f"import error: {exc}"
    from ...llm_serving.local_lora import LocalLoRA
    try:
        backend = LocalLoRA(base_model=base_model, adapter_path=adapter_path)
        ok = backend._try_load()
    except Exception as exc:
        return None, f"load error: {exc}"
    if not ok:
        return None, "adapter failed to load"
    prompt = json.dumps({"feature_view": {"dominant_signal": "tampered"}})
    return (lambda: backend.generate_json(prompt)), ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=200,
                    help="K=1 sample count (K=3 multiplies by 3)")
    ap.add_argument("--out", default="code/results/ijcip_latency")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. heuristic fallback
    fn = _build_call_heuristic()
    for k_label, k in [("k=1", 1), ("k=3", 3)]:
        stats = _benchmark(lambda: [fn() for _ in range(k)][-1], args.n_calls)
        rows.append({"backend": "heuristic_fallback", "k": k_label,
                      "status": "ok", "reason": "", **stats})

    # 2. real LoRA backends (gracefully marked not-run when unavailable)
    real_specs = [
        ("qwen2_5_7b_lora",  "Qwen/Qwen2.5-7B-Instruct",  "DER_LORA_ADAPTER_QWEN7B"),
        ("llama3_1_8b_lora", "meta-llama/Llama-3.1-8B-Instruct", "DER_LORA_ADAPTER_LLAMA8B"),
    ]
    for label, base_model, env_var in real_specs:
        call, reason = _build_call_real_lora(base_model, env_var)
        if call is None:
            for k_label in ("k=1", "k=3"):
                rows.append({"backend": label, "k": k_label,
                              "status": "not_run", "reason": reason,
                              "n_calls": 0, "mean_ms": float("nan"),
                              "p50_ms": float("nan"), "p95_ms": float("nan"),
                              "p99_ms": float("nan"), "max_ms": float("nan"),
                              "throughput_hz": float("nan"),
                              "schema_failure_rate": float("nan")})
            continue
        for k_label, k in [("k=1", 1), ("k=3", 3)]:
            stats = _benchmark(lambda: [call() for _ in range(k)][-1],
                                  max(20, args.n_calls // 4))
            rows.append({"backend": label, "k": k_label,
                          "status": "ok", "reason": "", **stats})

    df = pd.DataFrame(rows)
    df.to_csv(out / "latency_metrics.csv", index=False)
    n_ok = (df["status"] == "ok").sum()
    n_not = (df["status"] == "not_run").sum()
    readme = ["# LLM latency micro-benchmark\n",
                f"- ok rows: {n_ok}",
                f"- not-run rows: {n_not}",
                "",
                "## Backends",
                "- heuristic_fallback --- bundled deterministic stub",
                "- qwen2_5_7b_lora    --- requires DER_LORA_ADAPTER_QWEN7B env var",
                "- llama3_1_8b_lora   --- requires DER_LORA_ADAPTER_LLAMA8B env var"]
    if n_not:
        readme.append("\n## Not-run reasons")
        for r in rows:
            if r["status"] == "not_run":
                readme.append(f"- {r['backend']} ({r['k']}): {r['reason']}")
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote {n_ok} ok rows; {n_not} not-run rows")


if __name__ == "__main__":
    main()
