"""Phase-3 smoke: one end-to-end incident with the REAL QLoRA model in the loop.

Runs a full harness scenario with the genuine Qwen or Llama LoRA Hypothesis
Agent under DER_LLM_STRICT=1 (any heuristic substitution aborts the run) and
verifies afterwards, from the written traces, that every LLM call was served
by the intended adapter.

Usage (from the 2026_DER-SecAgent repo root):
    python3 -m code.evaluation.llm_in_loop.smoke_real_incident --backend qwen
    python3 -m code.evaluation.llm_in_loop.smoke_real_incident --backend llama
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from code.llm_serving import model_paths as _MP

REVISION_TAG = "ijcip_revision_r1r2_20260805"

BACKENDS = {
    "qwen": {
        "label": "qwen2_5_7b_lora",
        "base": _MP.QWEN_BASE,
        "adapter": "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter",
    },
    "llama": {
        "label": "llama3_1_8b_lora",
        "base": _MP.LLAMA_BASE,
        "adapter": "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--scenario",
                    default="code/simulation/scenarios/ieee13_fdi_inverter/config.yaml")
    ap.add_argument("--invoke-interval-s", type=float, default=600.0)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    args = ap.parse_args()

    spec = BACKENDS[args.backend]
    os.environ["DER_LLM_STRICT"] = "1"

    import yaml

    from ...llm_serving.local_lora import configure_default
    from ...Multi_AI_Agent.adapter import DERSecAgentDetector
    from ...simulation.feeder import StubFeeder
    from ...simulation.harness import run_scenario

    llm = configure_default(spec["base"], spec["adapter"],
                            max_new_tokens=args.max_new_tokens)

    cfg = yaml.safe_load(Path(args.scenario).read_text())
    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"])
    detector = DERSecAgentDetector(llm_invoke_interval_s=args.invoke_interval_s)

    out_root = f"code/results/{REVISION_TAG}/smoke/{spec['label']}"
    t0 = time.time()
    out = run_scenario(args.scenario, detector, seed=0, out_root=out_root,
                       feeder=feeder,
                       extra_manifest={
                           "llm_backend": spec["label"],
                           "llm_base_model": spec["base"],
                           "llm_adapter": spec["adapter"],
                           "llm_adapter_sha": llm.adapter_sha(),
                           "llm_model_config_hash": llm.model_config_hash(),
                           "llm_strict": True,
                           "llm_invoke_interval_s": args.invoke_interval_s,
                       })
    wall_s = time.time() - t0

    # ---- Post-run assertions: prove the real backend served every call ----
    assert llm.n_calls_real > 0, "FATAL: no real LLM calls were made"
    assert llm.n_calls_fallback == 0, "FATAL: heuristic fallback was used"

    lines = [json.loads(l) for l in (out / "decisions.jsonl").read_text().splitlines()]
    traces = [tr for l in lines
              for tr in ((l.get("decision_trace") or {}).get("llm_traces") or [])]
    assert traces, "FATAL: no llm_traces recorded in decisions.jsonl"
    bad = [tr for tr in traces if tr["backend"] != "real_lora"
           or tr["fallback_reason"] is not None
           or tr["adapter_sha"] != llm.adapter_sha()]
    assert not bad, f"FATAL: {len(bad)} trace(s) not served by the intended adapter"

    summary = {
        "backend": spec["label"],
        "adapter_sha": llm.adapter_sha(),
        "substrate": llm._substrate,
        "scenario": cfg["name"],
        "n_real_calls": llm.n_calls_real,
        "n_fallback_calls": llm.n_calls_fallback,
        "n_traces_in_log": len(traces),
        "per_call_latency_ms": {
            "mean": sum(t["latency_ms"] for t in traces) / len(traces),
            "max": max(t["latency_ms"] for t in traces),
        },
        "parse_ok_rate": sum(1 for t in traces if t["parse_ok"]) / len(traces),
        "wall_clock_s": wall_s,
        "run_dir": str(out),
    }
    (Path(out) / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
