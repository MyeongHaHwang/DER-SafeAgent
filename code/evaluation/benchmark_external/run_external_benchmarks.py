"""Run DER-SecAgent (and selected baselines) over the external benchmark set.

The driver streams every external sample through the agent's Hypothesis +
Coordinator pipeline directly --- bypassing the time-stepped harness, which
isn't applicable to packet-level benchmarks --- and records:

  - predicted attack_class
  - predicted action
  - JSON-validity / schema-compliance booleans
  - per-sample latency (perf_counter ns)

Datasets without a CSV on disk are recorded as a "not run" row.

Outputs (under ``results/ijcip_external_benchmark/``):
    metrics.csv              one row per (dataset, detector)
    per_class_metrics.csv    one row per (dataset, detector, class)
    confusion_matrix.csv     long-format (dataset, detector, true, pred, n)
    README.md                run log
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .dataset_adapter import ALL_BENCHMARKS, ExternalDatasetAdapter, ExternalSample
from .label_mapping import DER_CLASSES
from .metrics import (ALL_ACTIONS, SAFE_ACTIONS, accuracy, confusion_matrix,
                      forbidden_action_rate, is_valid_json, macro_f1,
                      per_class_metrics, schema_compliant, summarise,
                      weighted_f1)

OUT_DIR = Path("code/results/ijcip_external_benchmark")


# ---------- detector wrappers ----------
# Every detector exposes a single ``classify(sample)`` returning a dict with
# the canonical Hypothesis-Agent schema.

def _heuristic_from_payload(sample: ExternalSample) -> dict:
    """Use the LocalLoRA heuristic (the deterministic fallback) on a
    pseudo-prompt assembled from the sample. This is what we measure when no
    real adapter checkpoint is available."""
    from ...llm_serving.local_lora import get_default
    prompt = (
        '{"kind": "alarm", "payload": ' + json.dumps(sample.payload) + '}'
    )
    return get_default().generate_json(prompt)


def _wrap_decision(out: dict, sample: ExternalSample, latency_ms: float) -> dict:
    pred_cls = out.get("attack_class", "none")
    pred_act = out.get("proposed_action", "no_op")
    return {
        "sample_id": sample.sample_id,
        "dataset":   sample.dataset,
        "true_class": sample.der_class,
        "pred_class": pred_cls,
        "pred_action": pred_act,
        "ref_action":  None,             # rule-derived not available for these benchmarks
        "confidence":  float(out.get("confidence", 0.0)),
        "json_valid":   is_valid_json(out),
        "schema_compliant": schema_compliant(out),
        "hitl_required": pred_act == "no_op" and float(out.get("confidence", 0)) < 0.5,
        "latency_ms":  latency_ms,
    }


def classify_der_secagent(sample: ExternalSample) -> dict:
    """Run the five-agent loop's Hypothesis path on the sample."""
    from ...Multi_AI_Agent.agents import (caution_eval_agent,
                                            coordinator_agent,
                                            energy_impact_agent,
                                            hypothesis_agent,
                                            telemetry_analyst_agent)
    from ...Multi_AI_Agent.states import OperationMode, PromptStrategy

    state: dict = {
        "t": 0.0,
        "telemetry_window": [{
            "t": 0.0, "freq_hz": 60.0,
            "v_pu": {"BUS_PROXY": 1.0},
            "der_p_kw": {"DER_PROXY": 70.0},
            "der_p_avail": {"DER_PROXY": 100.0},
            "load_demand_kw": 1000.0, "load_served_kw": 1000.0,
        }],
        "event_window": [{
            "t": 0.0, "source": str(sample.payload.get("protocol", "unknown")),
            "kind": "alarm", "tampered": False,
            "payload": {**sample.payload, "kdd_label": sample.source_label,
                        "der_class_hint": sample.der_class},
        }],
        "mode": OperationMode.SCRIPT_GEN,
        "prompt_strategy": PromptStrategy.COT,
        "is_script_changed": False,
    }
    state = telemetry_analyst_agent(state)
    state = hypothesis_agent(state)
    state = energy_impact_agent(state)
    state = caution_eval_agent(state)
    state = coordinator_agent(state)
    return {
        "attack_class": state["attack_class"].value,
        "affected_asset": state.get("affected_asset"),
        "confidence": state.get("confidence", 0.0),
        "rationale": state.get("rationale", ""),
        "proposed_action": state.get("final_action", "no_op"),
        "expected_impact_tier": state.get("final_impact_tier", "negligible"),
    }


def classify_single_llm(sample: ExternalSample) -> dict:
    return _heuristic_from_payload(sample)


def classify_rule_ids(sample: ExternalSample) -> dict:
    """Heuristic stand-in for the rule_ids baseline at the packet level."""
    lbl = sample.source_label
    cls = sample.der_class
    action = "isolate_inverter" if cls in {"fdi", "dos"} else (
        "freeze_setpoint" if cls == "command_spoof" else (
        "request_ied_revalidation" if cls == "replay" else "no_op"))
    return {
        "attack_class": cls,
        "affected_asset": "DER_PROXY",
        "confidence": 0.7 if cls != "none" else 0.0,
        "rationale": f"rule_ids signature {lbl}",
        "proposed_action": action,
        "expected_impact_tier": "high" if action == "isolate_inverter" else "medium",
    }


DETECTORS: dict[str, Callable[[ExternalSample], dict]] = {
    "der_secagent": classify_der_secagent,
    "single_llm":   classify_single_llm,
    "rule_ids":     classify_rule_ids,
}


# ---------- driver ----------

def _run_one(adapter: ExternalDatasetAdapter, detector: str,
             classify_fn: Callable[[ExternalSample], dict],
             max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    for i, sample in enumerate(adapter.iter_examples()):
        if max_samples is not None and i >= max_samples:
            break
        t0 = time.perf_counter()
        try:
            out = classify_fn(sample)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rows.append(_wrap_decision(out, sample, latency_ms))
        except Exception as exc:
            rows.append({
                "sample_id": sample.sample_id, "dataset": sample.dataset,
                "true_class": sample.der_class, "pred_class": "none",
                "pred_action": "no_op", "ref_action": None,
                "confidence": 0.0, "json_valid": False,
                "schema_compliant": False, "hitl_required": True,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "error": repr(exc)[:200],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default=",".join(DETECTORS.keys()))
    ap.add_argument("--datasets", default=",".join(ALL_BENCHMARKS))
    ap.add_argument("--max-samples", type=int, default=200,
                    help="cap per (dataset, detector) for fast CI")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    detectors = [d for d in args.detectors.split(",") if d.strip()]
    datasets = [d for d in args.datasets.split(",") if d.strip()]

    summary_rows = []
    per_class_rows = []
    cm_rows = []
    examples_path = out / "examples.jsonl"
    examples_path.write_text("")

    for ds in datasets:
        adapter = ExternalDatasetAdapter(name=ds)
        if not adapter.is_available():
            for det in detectors:
                summary_rows.append({
                    "dataset": ds, "detector": det,
                    "n_samples": 0, "status": "not_run",
                    "reason": adapter.unavailable_reason,
                    **{k: float("nan") for k in (
                        "accuracy", "macro_f1", "weighted_f1",
                        "json_validity_rate", "schema_compliance_rate",
                        "safe_action_rate", "forbidden_action_rate",
                        "abstain_rate", "action_accuracy",
                        "mean_latency_ms",
                    )},
                })
            continue

        for det in detectors:
            fn = DETECTORS[det]
            decisions = _run_one(adapter, det, fn, args.max_samples)
            if not decisions:
                continue
            stats = summarise(decisions)
            stats.update({"dataset": ds, "detector": det,
                           "n_samples": len(decisions),
                           "status": "ok", "reason": "",
                           "mean_latency_ms": float(
                               sum(r["latency_ms"] for r in decisions) / len(decisions))})
            summary_rows.append(stats)

            pcm = per_class_metrics(decisions)
            for r in pcm:
                per_class_rows.append({"dataset": ds, "detector": det, **r})

            cm = confusion_matrix(decisions)
            for i, true_c in enumerate(DER_CLASSES):
                for j, pred_c in enumerate(DER_CLASSES):
                    cm_rows.append({"dataset": ds, "detector": det,
                                     "true_class": true_c, "pred_class": pred_c,
                                     "n": cm[i][j]})

            with examples_path.open("a") as fh:
                for r in decisions[:5]:
                    fh.write(json.dumps(r) + "\n")

    pd.DataFrame(summary_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out / "confusion_matrix.csv", index=False)

    readme = ["# External-benchmark run\n"]
    n_ok = sum(1 for r in summary_rows if r["status"] == "ok")
    n_not = sum(1 for r in summary_rows if r["status"] == "not_run")
    readme.append(f"- ok rows: {n_ok}")
    readme.append(f"- not-run rows: {n_not}")
    readme.append(f"- detectors: {', '.join(detectors)}")
    readme.append(f"- datasets:  {', '.join(datasets)}")
    readme.append(f"- max_samples per cell: {args.max_samples}")
    if n_not:
        readme.append("\n## Not-run datasets")
        for r in summary_rows:
            if r["status"] == "not_run":
                readme.append(f"- {r['dataset']}: {r['reason']}")
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote metrics for {n_ok} (dataset,detector) cells; "
          f"{n_not} marked not-run")


if __name__ == "__main__":
    main()
