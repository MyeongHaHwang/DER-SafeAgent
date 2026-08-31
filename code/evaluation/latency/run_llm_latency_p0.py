"""IJCIP P0 LLM-latency micro-benchmark.

Measures end-to-end Hypothesis-Agent latency for five backend
configurations:

    heuristic_fallback       (always available)
    qwen2_5_7b_base          (DER_BASE_QWEN7B)
    qwen2_5_7b_lora          (DER_BASE_QWEN7B + DER_LORA_ADAPTER_QWEN7B)
    llama3_1_8b_base         (DER_BASE_LLAMA8B)
    llama3_1_8b_lora         (DER_BASE_LLAMA8B + DER_LORA_ADAPTER_LLAMA8B)

Each backend is exercised under K=1 and K=3 self-consistency. For
every (backend, K) cell we record::

    backend_status      ok | not_run | failed
    not_run_reason      empty when status='ok'
    n_calls
    mean_ms / p50_ms / p95_ms / p99_ms / max_ms
    throughput_hz
    schema_failure_rate
    json_valid_rate
    gpu_mem_alloc_mb    (CUDA only)
    cpu_mem_rss_mb

When a real model is unavailable, the row is recorded with
``status='not_run'`` and a reason --- *no* numbers are fabricated.
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


# ---- helpers ---------------------------------------------------------------

def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = int(round(q * (len(s) - 1)))
    return s[k]


def _gpu_mem_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    return float("nan")


def _cpu_mem_mb() -> float:
    try:
        import resource  # POSIX
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(rss_kb) / 1024.0
    except Exception:
        return float("nan")


def _benchmark(call: Callable[[], dict], n: int) -> dict:
    latencies: list[float] = []
    schema_fail = 0
    json_fail = 0
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            out = call()
            ok_dict = isinstance(out, dict)
            if not ok_dict:
                json_fail += 1
            if not (ok_dict and "attack_class" in out):
                schema_fail += 1
        except Exception:
            json_fail += 1
            schema_fail += 1
        latencies.append((time.perf_counter() - t0) * 1000.0)
    total = sum(latencies)
    return {
        "n_calls":             n,
        "mean_ms":             statistics.mean(latencies) if latencies else float("nan"),
        "p50_ms":              _quantile(latencies, 0.50),
        "p95_ms":              _quantile(latencies, 0.95),
        "p99_ms":              _quantile(latencies, 0.99),
        "max_ms":              max(latencies) if latencies else float("nan"),
        "throughput_hz":       (1000.0 * n / total) if total > 0 else 0.0,
        "schema_failure_rate": schema_fail / n if n else float("nan"),
        "json_valid_rate":     1.0 - (json_fail / n) if n else float("nan"),
        "gpu_mem_alloc_mb":    _gpu_mem_mb(),
        "cpu_mem_rss_mb":      _cpu_mem_mb(),
    }


# ---- representative input samples -----------------------------------------

REPRESENTATIVE_PROMPTS = [
    json.dumps({"feature_view": {"dominant_signal": "tampered",
                                    "n_tampered_events": 3,
                                    "severity_score": 0.8},
                  "telemetry_window": [],
                  "event_window": [
                      {"kind": "telemetry", "tampered": True,
                        "payload": {"asset": "INV_634", "p_kw": 200}}
                  ],
                  "memory": "(none)"}),
    json.dumps({"feature_view": {"dominant_signal": "command",
                                    "n_command_events": 1,
                                    "severity_score": 0.7},
                  "telemetry_window": [],
                  "event_window": [
                      {"kind": "command", "tampered": True,
                        "payload": {"asset": "INV_634", "type": "setpoint",
                                      "p_kw": 25, "from": "DERMS"}}
                  ],
                  "memory": "(none)"}),
    json.dumps({"feature_view": {"dominant_signal": "freeze",
                                    "persistent_freeze": True,
                                    "severity_score": 0.78},
                  "telemetry_window": [], "event_window": [],
                  "memory": "(none)"}),
]


def _next_prompt(i: int) -> str:
    return REPRESENTATIVE_PROMPTS[i % len(REPRESENTATIVE_PROMPTS)]


# ---- backend builders ------------------------------------------------------

def _build_heuristic() -> Callable[[], dict]:
    from ...llm_serving.local_lora import get_default
    backend = get_default()
    counter = {"i": 0}
    def call() -> dict:
        out = backend.generate_json(_next_prompt(counter["i"]))
        counter["i"] += 1
        return out
    return call


def _build_real(base_model_env: str, adapter_env: str | None
                ) -> tuple[Callable[[], dict] | None, str]:
    base = os.environ.get(base_model_env, "")
    if not base:
        return None, f"env {base_model_env} unset"
    adapter = os.environ.get(adapter_env, "") if adapter_env else ""
    try:
        import torch  # noqa
        from peft import PeftModel  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
    except ImportError as exc:
        return None, f"import error: {exc}"

    from ...llm_serving.local_lora import LocalLoRA
    try:
        backend = LocalLoRA(base_model=base,
                              adapter_path=adapter or None)
        loaded = backend._try_load()
    except Exception as exc:
        return None, f"load error: {exc}"
    if not loaded:
        if not adapter:
            return None, "base path missing or unloadable on this host"
        return None, "adapter failed to load"

    counter = {"i": 0}
    def call() -> dict:
        out = backend.generate_json(_next_prompt(counter["i"]))
        counter["i"] += 1
        return out
    return call, ""


# ---- driver ----------------------------------------------------------------

BACKENDS = [
    ("heuristic_fallback", lambda: (_build_heuristic(), "")),
    ("qwen2_5_7b_base",    lambda: _build_real("DER_BASE_QWEN7B",  None)),
    ("qwen2_5_7b_lora",    lambda: _build_real("DER_BASE_QWEN7B",
                                                  "DER_LORA_ADAPTER_QWEN7B")),
    ("llama3_1_8b_base",   lambda: _build_real("DER_BASE_LLAMA8B", None)),
    ("llama3_1_8b_lora",   lambda: _build_real("DER_BASE_LLAMA8B",
                                                  "DER_LORA_ADAPTER_LLAMA8B")),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=100)
    ap.add_argument("--out", default="code/results/ijcip_latency")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    samples_log = out / "latency_samples.jsonl"
    samples_log.write_text("")

    for label, factory in BACKENDS:
        call, reason = factory()
        for k_label, k in [("k=1", 1), ("k=3", 3)]:
            if call is None:
                rows.append({"backend": label, "k": k_label,
                              "backend_status": "not_run",
                              "not_run_reason": reason or "n/a",
                              "n_calls": 0,
                              **{c: float("nan") for c in (
                                  "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                                  "max_ms", "throughput_hz",
                                  "schema_failure_rate", "json_valid_rate",
                                  "gpu_mem_alloc_mb", "cpu_mem_rss_mb")},
                              })
                continue
            n = args.n_calls if label == "heuristic_fallback" else max(20, args.n_calls // 4)
            try:
                stats = _benchmark(lambda: [call() for _ in range(k)][-1], n)
                stats.update({"backend": label, "k": k_label,
                                "backend_status": "ok", "not_run_reason": ""})
                rows.append(stats)
                with samples_log.open("a") as fh:
                    fh.write(json.dumps({"backend": label, "k": k_label,
                                            **stats}) + "\n")
            except Exception as exc:
                rows.append({"backend": label, "k": k_label,
                              "backend_status": "failed",
                              "not_run_reason": repr(exc)[:200],
                              "n_calls": 0,
                              **{c: float("nan") for c in (
                                  "mean_ms", "p50_ms", "p95_ms", "p99_ms",
                                  "max_ms", "throughput_hz",
                                  "schema_failure_rate", "json_valid_rate",
                                  "gpu_mem_alloc_mb", "cpu_mem_rss_mb")},
                              })

    df = pd.DataFrame(rows)
    df.to_csv(out / "latency_metrics.csv", index=False)

    n_ok  = (df["backend_status"] == "ok").sum()
    n_not = (df["backend_status"] == "not_run").sum()
    n_fail = (df["backend_status"] == "failed").sum()
    readme = ["# IJCIP P0 LLM-latency benchmark\n",
                f"- ok rows  : {n_ok}",
                f"- not-run  : {n_not}",
                f"- failed   : {n_fail}",
                "",
                "## Backends",
                "- `heuristic_fallback` --- bundled deterministic stub",
                "- `qwen2_5_7b_base`  --- needs `DER_BASE_QWEN7B`",
                "- `qwen2_5_7b_lora`  --- needs `DER_BASE_QWEN7B` + `DER_LORA_ADAPTER_QWEN7B`",
                "- `llama3_1_8b_base` --- needs `DER_BASE_LLAMA8B`",
                "- `llama3_1_8b_lora` --- needs `DER_BASE_LLAMA8B` + `DER_LORA_ADAPTER_LLAMA8B`",
                "",
                "Real-model rows are recorded as `not_run` whenever the",
                "corresponding env variables are unset or the weights cannot",
                "be loaded. No latency numbers are fabricated."]
    if n_not:
        readme.append("\n## Not-run reasons")
        for r in rows:
            if r["backend_status"] == "not_run":
                readme.append(f"- {r['backend']} ({r['k']}): {r['not_run_reason']}")
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote latency metrics: ok={n_ok} not_run={n_not} failed={n_fail}")


if __name__ == "__main__":
    main()
