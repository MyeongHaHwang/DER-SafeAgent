"""LLM-in-the-loop cyber-physical evaluation over the expanded scenario matrix.

System configurations (frozen names, see paper/revision_experiment_plan.md §2):

  D0   deterministic_energy_policy   no LLM, fast-path baseline
  H0   der_secagent_heuristic_k3     legacy scripted-hypothesis backbone (labelled)
  RID  rule_ids                      signature IDS baseline
  PM   prior_mas_heuristic           2-agent prior baseline (heuristic backbone)
  ORC  oracle_class_policy           ground-truth upper bound
  B1   bare_qwen_lora                REAL Qwen2.5-7B+LoRA, no safety stack
  Q1   der_secagent_qwen_lora_k1     REAL Qwen LoRA, K=1, full safety stack
  L1   der_secagent_llama_lora_k1    REAL Llama-3.1-8B+LoRA, K=1, full stack
  Q3   der_secagent_qwen_lora_k3     REAL Qwen LoRA, K=3, full stack (subset)
  L3   der_secagent_llama_lora_k3    REAL Llama LoRA, K=3, full stack (subset)

Every configuration whose name contains a model identifier runs with
DER_LLM_STRICT=1: a single heuristic-fallback call aborts the run. Post-run
assertions additionally verify, from the written decisions.jsonl, that every
recorded LLM call was served by the intended adapter (backend == real_lora,
matching adapter SHA, fallback_reason None).

Runs are resumable: a run directory with a manifest.json is never re-executed
or overwritten.

Usage examples (repo root):
  python3 -m code.evaluation.llm_in_loop.run_llm_in_loop --systems D0,H0,RID --scenarios all
  python3 -m code.evaluation.llm_in_loop.run_llm_in_loop --systems Q1 --scenarios all
  python3 -m code.evaluation.llm_in_loop.run_llm_in_loop --systems Q3,L3 --scenarios k3_subset
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..caution_metrics import aggregate_run as caution_aggregate
from ..detection_metrics import aggregate_run as detection_aggregate, macro_f1
from ..physical_curves import sweep
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario
from code.llm_serving import model_paths as _MP

REVISION_TAG = "ijcip_revision_r1r2_20260805"
OUT_ROOT = Path("code/results") / REVISION_TAG / "llm_in_loop"
MANIFEST_CSV = Path("code/configs") / REVISION_TAG / "scenario_manifest.csv"

QWEN_BASE = _MP.QWEN_BASE
QWEN_ADAPTER = "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter"
LLAMA_BASE = _MP.LLAMA_BASE
LLAMA_ADAPTER = "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter"

# Incident-level invocation period for real-model runs (see adapter.py). One
# LLM consultation per incident: the trigger-only serving posture the paper
# describes. The deterministic layers still run on every tick.
INVOKE_INTERVAL_S = 600.0
# The target JSON is ~70 tokens; 128 leaves headroom without letting a
# non-terminating generation burn the whole CPU budget.
MAX_NEW_TOKENS = 128

# ---------------------------------------------------------------------------
# PRE-REGISTERED SCENARIO SUBSETS (declared before any real-LLM cyber-physical
# result was produced; recorded here and in the frozen scenario matrix).
#
# Rationale: on the available substrate the GPU is unusable (driver not
# configured) and CPU inference of a 7-8B model costs minutes per call, so the
# full 25-configuration matrix x 2 models x 2 K settings is not reachable in
# the revision window. LLM_SUBSET spans every attack type, all three magnitude
# levels, all three duration levels, both feeders, the fleet-level target, both
# non-nominal load levels, both non-nominal penetration levels, and the benign
# control -- 12 configurations, which is the minimum unique-configuration count
# required by the revision brief. K3_SUBSET is a nested subset of it.
LLM_SUBSET = [
    "rv13_fdi_med_medium_nom_medpen_inv",          # fdi, centre point
    "rv13_fdi_high_medium_nom_medpen_inv",         # fdi, high magnitude
    "rv13_spoof_med_medium_nom_medpen_inv",        # spoof, centre point
    "rv13_spoof_low_medium_nom_medpen_inv",        # spoof, low magnitude
    "rv13_spoof_high_medium_nom_medpen_inv",       # spoof, high magnitude
    "rv13_spoof_med_short_nom_medpen_inv",         # short duration
    "rv13_spoof_med_persistent_nom_medpen_inv",    # persistent duration
    "rv13_spoof_med_medium_peak_medpen_inv",       # peak load
    "rv13_spoof_med_medium_nom_highpen_inv",       # high DER penetration
    "rv13_replay_med_medium_nom_medpen_inv",       # replay
    "rv13_dos_med_medium_nom_medpen_inv",          # dos
    "rv34_spoof_high_medium_nom_medpen_fleet",     # ieee34, fleet-level target
]

# K=3 subset: 6 configurations nested inside LLM_SUBSET, one per attack type
# plus the two extreme operating conditions.
K3_SUBSET = [
    "rv13_fdi_med_medium_nom_medpen_inv",
    "rv13_spoof_med_medium_nom_medpen_inv",
    "rv13_replay_med_medium_nom_medpen_inv",
    "rv13_dos_med_medium_nom_medpen_inv",
    "rv13_spoof_med_medium_peak_medpen_inv",
    "rv34_spoof_high_medium_nom_medpen_fleet",
]


def _mk_der_secagent(name: str, k: int):
    """Full DER-SecAgent safety stack with a real LoRA Hypothesis Agent.

    ``compact_prompt=True`` serves the adapter the fine-tuning-distribution
    prompt (see agents._compact_windows); the long published prompt is ~2.4k
    tokens, which is both out-of-distribution for these adapters and, on the
    CPU substrate available here, minutes per call.
    """
    from ...Multi_AI_Agent.adapter import DERSecAgentDetector
    det = DERSecAgentDetector(
        ablation="full", k_setting=k, compact_prompt=True,
        llm_invoke_interval_s=INVOKE_INTERVAL_S,
    )
    det.name = name
    return det


def _mk_bare_llm(name: str):
    from ...baselines.single_llm.runner import SingleLLM
    det = SingleLLM(invoke_interval_s=INVOKE_INTERVAL_S)
    det.name = name
    return det


def _mk_simple(spec: str, name: str | None = None):
    mod, cls = spec.split(":")
    det = getattr(importlib.import_module(mod), cls)()
    if name:
        det.name = name
    return det


SYSTEMS: dict[str, dict] = {
    "D0": {"label": "deterministic_energy_policy", "llm": None,
           "make": lambda: _mk_simple("code.baselines.deterministic_energy_policy.adapter:DeterministicEnergyPolicy")},
    "H0": {"label": "der_secagent_heuristic_k3", "llm": "heuristic",
           "make": lambda: _mk_heuristic_der_secagent()},
    "RID": {"label": "rule_ids", "llm": None,
            "make": lambda: _mk_simple("code.baselines.rule_ids.adapter:RuleIDS")},
    "PM": {"label": "prior_mas_heuristic", "llm": "heuristic",
           "make": lambda: _mk_simple("code.baselines.prior_mas.adapter:PriorMAS", "prior_mas_heuristic")},
    "ORC": {"label": "oracle_class_policy", "llm": None,
            "make": lambda: _mk_simple("code.baselines.oracle_class_policy.adapter:OracleClassPolicy")},
    "B1": {"label": "bare_qwen_lora", "llm": "qwen",
           "make": lambda: _mk_bare_llm("bare_qwen_lora")},
    "Q1": {"label": "der_secagent_qwen_lora_k1", "llm": "qwen",
           "make": lambda: _mk_der_secagent("der_secagent_qwen_lora_k1", k=1)},
    "L1": {"label": "der_secagent_llama_lora_k1", "llm": "llama",
           "make": lambda: _mk_der_secagent("der_secagent_llama_lora_k1", k=1)},
    "Q3": {"label": "der_secagent_qwen_lora_k3", "llm": "qwen",
           "make": lambda: _mk_der_secagent("der_secagent_qwen_lora_k3", k=3)},
    "L3": {"label": "der_secagent_llama_lora_k3", "llm": "llama",
           "make": lambda: _mk_der_secagent("der_secagent_llama_lora_k3", k=3)},
}


def _mk_heuristic_der_secagent():
    from ...Multi_AI_Agent.adapter import DERSecAgentDetector
    det = DERSecAgentDetector()          # published configuration: K=3, no gate
    det.name = "der_secagent_heuristic_k3"
    return det


def _setup_llm(kind: str | None):
    """Configure the LLM singleton for this system; returns (llm, manifest_block)."""
    from ...llm_serving import local_lora
    if kind in (None, "heuristic"):
        os.environ.pop("DER_LLM_STRICT", None)
        os.environ.pop("DER_LORA_ADAPTER", None)
        local_lora.reset_default()
        return None, {"llm_backend": kind or "none"}
    base, adapter = (QWEN_BASE, QWEN_ADAPTER) if kind == "qwen" else (LLAMA_BASE, LLAMA_ADAPTER)
    os.environ["DER_LLM_STRICT"] = "1"
    llm = local_lora.configure_default(base, adapter, max_new_tokens=MAX_NEW_TOKENS)
    if not llm._try_load():
        raise RuntimeError(
            f"FATAL: real backend {kind} could not be loaded: {llm._load_fail_reason}. "
            "Refusing to run an LLM-labelled configuration without the model.")
    return llm, {
        "llm_backend": f"{kind}_lora_real",
        "llm_base_model": base,
        "llm_adapter": adapter,
        "llm_adapter_sha": llm.adapter_sha(),
        "llm_model_config_hash": llm.model_config_hash(),
        "llm_substrate": llm._substrate,
        "llm_strict": True,
        "llm_invoke_interval_s": INVOKE_INTERVAL_S,
        "llm_max_new_tokens": MAX_NEW_TOKENS,
    }


def _assert_real_llm_run(run_dir: Path, expected_sha: str) -> dict:
    """Verify from the written traces that only the intended adapter served calls."""
    lines = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
    traces = [tr for l in lines
              for tr in ((l.get("decision_trace") or {}).get("llm_traces") or [])]
    n_bad = sum(1 for tr in traces if tr["backend"] != "real_lora"
                or tr["fallback_reason"] is not None
                or tr["adapter_sha"] != expected_sha)
    if n_bad:
        raise AssertionError(
            f"{run_dir}: {n_bad}/{len(traces)} LLM traces not served by the "
            f"intended adapter {expected_sha}")
    if not traces:
        # A run with zero consultations is legitimate when the trigger gate
        # never fired (e.g. the benign calibration configuration): the
        # deterministic layers ran and correctly found nothing to escalate.
        # What must never happen is a consultation served by something other
        # than the intended adapter, which is checked above. We record the
        # zero explicitly so it cannot be mistaken for a missing measurement.
        if any(json.loads(l).get("decision_trace") for l in
               (run_dir / "decisions.jsonl").read_text().splitlines()):
            raise AssertionError(
                f"{run_dir}: agent pipeline ran but recorded no LLM trace in an "
                f"LLM-labelled run")
        return {"n_llm_calls": 0, "parse_ok_rate": float("nan"),
                "mean_llm_latency_ms": float("nan"),
                "p95_llm_latency_ms": float("nan"),
                "llm_never_consulted": True}
    return {"n_llm_calls": len(traces),
            "parse_ok_rate": sum(t["parse_ok"] for t in traces) / len(traces),
            "mean_llm_latency_ms": float(np.mean([t["latency_ms"] for t in traces])),
            "p95_llm_latency_ms": float(np.quantile([t["latency_ms"] for t in traces], 0.95)),
            "llm_never_consulted": False}


def _trace_behaviour(run_dir: Path) -> dict:
    """Proposed-vs-final divergence and safety-layer interception counts."""
    lines = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
    dts = [l["decision_trace"] for l in lines if l.get("decision_trace")]
    n = len(dts)
    changed = sum(1 for d in dts
                  if d.get("proposed_action") and d.get("final_action")
                  and d["proposed_action"] != d["final_action"])
    hitl = sum(1 for d in dts if d.get("coordinator_decision") == "hitl_required")
    caution = sum(1 for d in dts if d.get("caution"))
    return {"n_decision_steps": n, "n_proposed_final_divergent": changed,
            "n_hitl_required": hitl, "n_caution_flags": caution}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True,
                    help="comma list of system ids, e.g. D0,H0,Q1")
    ap.add_argument("--scenarios", default="all",
                    help="'all', 'k3_subset', 'smoke', or comma list of scenario ids")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST_CSV)
    if args.scenarios == "all":
        chosen = man
    elif args.scenarios == "llm_subset":
        chosen = man[man.scenario_id.isin(LLM_SUBSET)]
    elif args.scenarios == "k3_subset":
        chosen = man[man.scenario_id.isin(K3_SUBSET)]
    elif args.scenarios == "smoke":
        chosen = man[man.scenario_id.isin(K3_SUBSET[:3])]
    else:
        chosen = man[man.scenario_id.isin(args.scenarios.split(","))]
    assert len(chosen) > 0, "no scenarios selected"

    summary_rows = []
    for sys_id in args.systems.split(","):
        spec = SYSTEMS[sys_id]
        llm, llm_manifest = _setup_llm(spec["llm"])
        runs_root = OUT_ROOT / spec["label"] / "runs"

        for _, row in chosen.iterrows():
            cfg_path = row.config_path
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            run_dir = runs_root / cfg["name"] / spec["label"] / f"seed{args.seed}"
            if (run_dir / "manifest.json").exists():
                print(f"[skip] {sys_id} {cfg['name']} (completed)")
            else:
                detector = spec["make"]()
                if sys_id == "ORC":
                    gt = cfg.get("ground_truth") or {}
                    detector.configure(
                        attack_class=gt.get("attack_class", "none"),
                        target=gt.get("affected_asset") or "",
                        start_s=float(gt.get("start_s", 0) or 0),
                        end_s=float(gt.get("end_s", 0) or 0),
                    )
                feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                    ders=cfg["ders"],
                                    base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
                t0 = time.time()
                run_scenario(cfg_path, detector, args.seed, out_root=str(runs_root),
                             feeder=feeder,
                             extra_manifest={**llm_manifest,
                                             "revision_tag": REVISION_TAG,
                                             "system_id": sys_id,
                                             "scenario_config_hash": row.configuration_hash,
                                             "evaluation_mode": "decision_quality"})
                print(f"[done] {sys_id} {cfg['name']} in {time.time()-t0:.1f}s")

            # per-run metrics
            rec = {"system": spec["label"], "system_id": sys_id,
                   "scenario": cfg["name"], "seed": args.seed,
                   "configuration_hash": row.configuration_hash,
                   "is_calibration_control": bool(row.is_calibration_control)}
            phys = sweep(run_dir, np.linspace(0.0, 1.0, 11))
            at05 = phys[phys["threshold"].between(0.49, 0.51)].iloc[0]
            rec.update({m: float(at05[m]) for m in
                        ("ens_kwh", "curt_kwh", "voltage_frac", "freq_dev_hz",
                         "ramp_violations")})
            det_df = detection_aggregate(run_dir)
            rec["macro_f1"] = macro_f1(det_df)
            rec.update(caution_aggregate(run_dir))
            rec.update(_trace_behaviour(run_dir))
            if llm is not None:
                rec.update(_assert_real_llm_run(run_dir, llm.adapter_sha()))
            summary_rows.append(rec)

    out_csv = OUT_ROOT / "llm_in_loop_summary.csv"
    df = pd.DataFrame(summary_rows)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        df = (pd.concat([old, df], ignore_index=True)
              .drop_duplicates(subset=["system", "scenario", "seed"], keep="last"))
    df.to_csv(out_csv, index=False)
    print(f"summary -> {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
