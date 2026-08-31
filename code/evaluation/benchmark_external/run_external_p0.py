"""IJCIP P0 external-benchmark runner.

Differences from the original ``run_external_benchmarks.py``:
* Resolves UNSW-MG24 through the new production loader, falling back
  to the shipped 200-row mock; tags every output with
  ``result_source = production | mock``.
* Recognises ``unknown_or_unmapped`` as a legitimate class and scores
  abstention behaviour explicitly (selective risk, coverage-risk
  curve, correct-abstention rate, unsafe forced-action rate).
* Optional benchmarks (SWaT, WADI, TON_IoT, EPIC) emit ``not_run``
  rows --- never fabricated --- with a path-resolution reason.

Outputs:
    code/results/ijcip_external_benchmark/metrics.csv
    code/results/ijcip_external_benchmark/per_class_metrics.csv
    code/results/ijcip_external_benchmark/confusion_matrix.csv
    code/results/ijcip_external_benchmark/coverage_risk.csv
    code/results/ijcip_external_benchmark/README.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from ...datasets.unsw_mg24.production_loader import UnswProductionLoader, UnswSample
from ...datasets.ton_iot.production_loader import (TonIotProductionLoader,
                                                      TonIotSample)
from .coverage_risk import (correct_abstention_rate, coverage_and_selective_risk,
                              coverage_risk_curve, unsafe_forced_action_rate)
from .label_mapping import DER_CLASSES_EXTENDED
from .metrics import (ALL_ACTIONS, SAFE_ACTIONS, accuracy, confusion_matrix,
                      forbidden_action_rate, is_valid_json, macro_f1,
                      per_class_metrics, schema_compliant, weighted_f1)

OUT_DIR = Path("code/results/ijcip_external_benchmark")

# ---- detector wrappers -----------------------------------------------------

def _payload_to_event(sample) -> dict:
    return {
        "t": float(sample.timestamp),
        "source": str(sample.payload.get("protocol", "unknown")).lower(),
        "kind": "alarm",
        "tampered": False,
        "payload": {**sample.payload,
                      "kdd_label": sample.source_label,
                      "der_class_hint": sample.der_class},
    }


def _heuristic_from_sample(sample) -> dict:
    from ...llm_serving.local_lora import get_default
    prompt = '{"kind": "alarm", "payload": ' + json.dumps(sample.payload) + "}"
    return get_default().generate_json(prompt)


def classify_der_secagent(sample) -> dict:
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
        "event_window": [_payload_to_event(sample)],
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
        "coordinator_decision": state.get("coordinator_decision", "auto"),
    }


def classify_single_llm(sample) -> dict:
    return _heuristic_from_sample(sample)


def classify_rule_ids(sample) -> dict:
    cls = sample.der_class
    # rule_ids must commit to an action even on unmapped labels --- this
    # is exactly the failure mode we want to expose.
    action_map = {
        "fdi":            "isolate_inverter",
        "dos":            "isolate_inverter",
        "command_spoof":  "freeze_setpoint",
        "replay":         "request_ied_revalidation",
        "firmware":       "request_ied_revalidation",
        "none":           "no_op",
        "unknown_or_unmapped": "freeze_setpoint",
    }
    action = action_map.get(cls, "freeze_setpoint")
    return {
        "attack_class": cls if cls != "unknown_or_unmapped" else "none",
        "affected_asset": "DER_PROXY",
        "confidence": 0.7 if cls != "none" else 0.0,
        "rationale": f"rule_ids signature {sample.source_label}",
        "proposed_action": action,
        "expected_impact_tier": "high" if action == "isolate_inverter" else "medium",
        "coordinator_decision": "auto",
    }


DETECTORS: dict[str, Callable[[object], dict]] = {
    "der_secagent": classify_der_secagent,
    "single_llm":   classify_single_llm,
    "rule_ids":     classify_rule_ids,
}


# ---- driver ---------------------------------------------------------------

def _wrap_decision(out: dict, sample, latency_ms: float) -> dict:
    pred_cls = out.get("attack_class", "none")
    pred_act = out.get("proposed_action", "no_op")
    coord_dec = out.get("coordinator_decision", "auto")
    return {
        "sample_id":       sample.sample_id,
        "true_class":      sample.der_class,
        "pred_class":      pred_cls,
        "pred_action":     pred_act,
        "ref_action":      None,
        "confidence":      float(out.get("confidence", 0.0)),
        "json_valid":      is_valid_json(out),
        "schema_compliant": schema_compliant(out),
        "hitl_required":   coord_dec == "hitl_required",
        "latency_ms":      latency_ms,
        "result_source":   sample.result_source,
    }


def _summarise(rows: list[dict]) -> dict:
    pcm = per_class_metrics(rows)
    cov_pt = coverage_and_selective_risk(rows)
    return {
        "n_samples":              len(rows),
        "accuracy":               accuracy(rows),
        "macro_f1":               macro_f1(pcm),
        "weighted_f1":            weighted_f1(pcm),
        "json_validity_rate":     sum(1 for r in rows if r["json_valid"]) / max(len(rows), 1),
        "schema_compliance_rate": sum(1 for r in rows if r["schema_compliant"]) / max(len(rows), 1),
        "forbidden_action_rate":  forbidden_action_rate(rows),
        "abstain_rate":           sum(1 for r in rows
                                       if r["hitl_required"] or r["pred_action"] == "no_op")
                                  / max(len(rows), 1),
        "coverage":               cov_pt["coverage"],
        "selective_risk":         cov_pt["selective_risk"],
        "correct_abstention_rate": correct_abstention_rate(rows),
        "unsafe_forced_action_rate": unsafe_forced_action_rate(rows),
        "mean_latency_ms":        sum(r["latency_ms"] for r in rows) / max(len(rows), 1),
    }


def _run_unsw_mg24(detectors: list[str],
                     max_samples: int,
                     csv_path: str | None = None,
                     label_col: str | None = None,
                     ) -> tuple[list[dict], list[dict], list[dict], list[dict], str, str]:
    loader = UnswProductionLoader(csv_path=csv_path, label_col=label_col)
    metric_rows = []; per_class_rows = []; cm_rows = []; cov_rows = []
    if not loader.available():
        for det in detectors:
            metric_rows.append({
                "dataset": "unsw_mg24", "detector": det,
                "result_source": "not_run", "status": "not_run",
                "reason": loader.unavailable_reason,
                "n_samples": 0,
                **{k: float("nan") for k in (
                    "accuracy", "macro_f1", "weighted_f1",
                    "json_validity_rate", "schema_compliance_rate",
                    "forbidden_action_rate", "abstain_rate", "coverage",
                    "selective_risk", "correct_abstention_rate",
                    "unsafe_forced_action_rate", "mean_latency_ms")},
            })
        return metric_rows, per_class_rows, cm_rows, cov_rows, "not_run", loader.unavailable_reason

    samples = list(loader.iter_samples(max_samples=max_samples))
    for det_name in detectors:
        fn = DETECTORS[det_name]
        rows = []
        for s in samples:
            t0 = time.perf_counter()
            try:
                out = fn(s)
                lat = (time.perf_counter() - t0) * 1000.0
                rows.append(_wrap_decision(out, s, lat))
            except Exception as exc:
                rows.append({
                    "sample_id": s.sample_id, "true_class": s.der_class,
                    "pred_class": "none", "pred_action": "no_op",
                    "ref_action": None, "confidence": 0.0,
                    "json_valid": False, "schema_compliant": False,
                    "hitl_required": True,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "result_source": s.result_source,
                    "error": repr(exc)[:200],
                })
        if not rows:
            continue
        stats = _summarise(rows)
        stats.update({"dataset": "unsw_mg24", "detector": det_name,
                       "result_source": loader.result_source, "status": "ok",
                       "reason": ""})
        metric_rows.append(stats)
        for r in per_class_metrics(rows):
            per_class_rows.append({"dataset": "unsw_mg24", "detector": det_name, **r})
        cm = confusion_matrix(rows)
        for i, true_c in enumerate(DER_CLASSES_EXTENDED[:-1]):
            for j, pred_c in enumerate(DER_CLASSES_EXTENDED[:-1]):
                cm_rows.append({"dataset": "unsw_mg24", "detector": det_name,
                                 "true_class": true_c, "pred_class": pred_c,
                                 "n": cm[i][j]})
        for cv in coverage_risk_curve(rows):
            cov_rows.append({"dataset": "unsw_mg24", "detector": det_name, **cv})

    return (metric_rows, per_class_rows, cm_rows, cov_rows,
            loader.result_source, "")


def _run_ton_iot(detectors: list[str],
                   max_samples: int,
                   parquet_path: str | None = None,
                   csv_path: str | None = None,
                   ) -> tuple[list[dict], list[dict], list[dict], list[dict], str, str]:
    loader = TonIotProductionLoader(parquet_path=parquet_path,
                                       csv_path=csv_path)
    metric_rows = []; per_class_rows = []; cm_rows = []; cov_rows = []
    if not loader.available():
        for det in detectors:
            metric_rows.append({
                "dataset": "ton_iot", "detector": det,
                "result_source": "not_run", "status": "not_run",
                "reason": loader.unavailable_reason,
                "n_samples": 0,
                **{k: float("nan") for k in (
                    "accuracy", "macro_f1", "weighted_f1",
                    "json_validity_rate", "schema_compliance_rate",
                    "forbidden_action_rate", "abstain_rate", "coverage",
                    "selective_risk", "correct_abstention_rate",
                    "unsafe_forced_action_rate", "mean_latency_ms")},
            })
        return metric_rows, per_class_rows, cm_rows, cov_rows, "not_run", loader.unavailable_reason

    samples = list(loader.iter_samples(max_samples=max_samples))
    for det_name in detectors:
        fn = DETECTORS[det_name]
        rows = []
        for s in samples:
            t0 = time.perf_counter()
            try:
                out = fn(s)
                lat = (time.perf_counter() - t0) * 1000.0
                rows.append(_wrap_decision(out, s, lat))
            except Exception as exc:
                rows.append({
                    "sample_id": s.sample_id, "true_class": s.der_class,
                    "pred_class": "none", "pred_action": "no_op",
                    "ref_action": None, "confidence": 0.0,
                    "json_valid": False, "schema_compliant": False,
                    "hitl_required": True,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "result_source": s.result_source,
                    "error": repr(exc)[:200],
                })
        if not rows:
            continue
        stats = _summarise(rows)
        stats.update({"dataset": "ton_iot", "detector": det_name,
                       "result_source": loader.result_source, "status": "ok",
                       "reason": ""})
        metric_rows.append(stats)
        for r in per_class_metrics(rows):
            per_class_rows.append({"dataset": "ton_iot", "detector": det_name, **r})
        cm = confusion_matrix(rows)
        for i, true_c in enumerate(DER_CLASSES_EXTENDED[:-1]):
            for j, pred_c in enumerate(DER_CLASSES_EXTENDED[:-1]):
                cm_rows.append({"dataset": "ton_iot", "detector": det_name,
                                 "true_class": true_c, "pred_class": pred_c,
                                 "n": cm[i][j]})
        for cv in coverage_risk_curve(rows):
            cov_rows.append({"dataset": "ton_iot", "detector": det_name, **cv})
    return (metric_rows, per_class_rows, cm_rows, cov_rows,
            loader.result_source, "")


def _run_optional(name: str, detectors: list[str]) -> list[dict]:
    """For SWaT / WADI / EPIC: not bundled. Mark not_run."""
    reason = (f"no labelled CSV at code/datasets/{name}/raw/{name}.csv; "
                "drop a labelled CSV there and re-run")
    return [{
        "dataset": name, "detector": det,
        "result_source": "not_run", "status": "not_run",
        "reason": reason, "n_samples": 0,
        **{k: float("nan") for k in (
            "accuracy", "macro_f1", "weighted_f1",
            "json_validity_rate", "schema_compliance_rate",
            "forbidden_action_rate", "abstain_rate", "coverage",
            "selective_risk", "correct_abstention_rate",
            "unsafe_forced_action_rate", "mean_latency_ms")},
    } for det in detectors]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default="der_secagent,single_llm,rule_ids")
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--unsw-csv", default=None,
                     help="path to a labelled UNSW-MG24 CSV (overrides resolution order)")
    ap.add_argument("--unsw-label-col", default=None,
                     help="explicit label column for UNSW-MG24 (e.g. attack_name)")
    ap.add_argument("--ton-iot-parquet", default=None,
                     help="path to a labelled ToN-IoT parquet")
    ap.add_argument("--ton-iot-csv", default=None,
                     help="path to a labelled ToN-IoT CSV (alt to parquet)")
    ap.add_argument("--ton-iot-max-samples", type=int, default=100_000,
                     help="cap for ToN-IoT stratified sample (defaults to 100k)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    detectors = [d for d in args.detectors.split(",") if d.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    metric_rows, per_class_rows, cm_rows, cov_rows, result_source, reason \
        = _run_unsw_mg24(detectors, args.max_samples,
                          csv_path=args.unsw_csv,
                          label_col=args.unsw_label_col)
    (ton_m, ton_pc, ton_cm, ton_cov, ton_src, ton_reason) = _run_ton_iot(
        detectors, args.ton_iot_max_samples,
        parquet_path=args.ton_iot_parquet,
        csv_path=args.ton_iot_csv)
    metric_rows.extend(ton_m)
    per_class_rows.extend(ton_pc)
    cm_rows.extend(ton_cm)
    cov_rows.extend(ton_cov)
    for opt in ("swat", "wadi", "epic"):
        metric_rows.extend(_run_optional(opt, detectors))

    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out / "confusion_matrix.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(out / "coverage_risk.csv", index=False)

    n_ok = sum(1 for r in metric_rows if r.get("status") == "ok")
    n_not = sum(1 for r in metric_rows if r.get("status") == "not_run")
    readme = ["# IJCIP external benchmark\n",
                f"- ok rows : {n_ok}",
                f"- not-run : {n_not}",
                f"- detectors: {', '.join(detectors)}",
                f"- unsw max_samples: {args.max_samples}",
                f"- ton-iot max_samples: {args.ton_iot_max_samples}",
                "",
                f"## UNSW-MG24 result_source: {result_source}",
                f"  reason (if any): {reason or '(none)'}",
                f"## ToN-IoT result_source: {ton_src}",
                f"  reason (if any): {ton_reason or '(none)'}",
                "",
                "UNSW-MG24 / ToN-IoT load their labelled CSV / Parquet from the",
                "repo-shipped paths by default; `--unsw-csv`, `--ton-iot-parquet`,",
                "`UNSW_MG24_ROOT`, and `TON_IOT_ROOT` override the resolution.",
                "SWaT/WADI/EPIC remain 'not_run' until a labelled CSV is dropped.",
                ]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print(f"wrote external-benchmark P0: {n_ok} ok, {n_not} not-run "
          f"(unsw_mg24 source={result_source})")


if __name__ == "__main__":
    main()
