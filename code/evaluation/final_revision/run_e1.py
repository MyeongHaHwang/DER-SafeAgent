"""E1: causal real-LLM-in-the-loop evaluation under two runtime-assurance modes.

Answers the question the previous revision could not: does the language model
have a causal path to the executed action, and does using that path change the
physical outcome?

Arms (see REPOSITORY_AUDIT.md §3):
  D0    deterministic fast path, no LLM
  H0    full stack, scripted hypothesis (legacy reference, not an LLM result)
  B0-Q  bare Qwen QLoRA, registry only, no shield
  S1-Q  single Qwen QLoRA + the same shield the full system uses
  M1-Q  full DER-SecAgent, real Qwen QLoRA, K=1
  M1-L  full DER-SecAgent, real Llama QLoRA, K=1
  ORC   ground-truth oracle, upper-bound reference

Each LLM arm is run under both ``legacy_class_override`` and
``safety_projection`` so the policy is a controlled variable.

Run: python3 -m code.evaluation.final_revision.run_e1 --arms D0,H0,M1-Q [--modes ...]
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

from ..physical_curves import sweep
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario
from code.llm_serving import model_paths as _MP

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "e1_causal"
MANIFEST = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")

QWEN = (_MP.QWEN_BASE,
        "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter")
LLAMA = (_MP.LLAMA_BASE,
         "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter")
INVOKE_S = 600.0
MAX_TOK = 128

ARMS = {
    "D0":   {"llm": None,    "kind": "deterministic"},
    "H0":   {"llm": "heur",  "kind": "full"},
    "ORC":  {"llm": None,    "kind": "oracle"},
    "B0-Q": {"llm": "qwen",  "kind": "bare"},
    "B0-L": {"llm": "llama", "kind": "bare"},
    "S1-Q": {"llm": "qwen",  "kind": "shielded_single"},
    "S1-L": {"llm": "llama", "kind": "shielded_single"},
    "M1-Q": {"llm": "qwen",  "kind": "full"},
    "M1-L": {"llm": "llama", "kind": "full"},
}


def _setup_llm(kind):
    from ...llm_serving import local_lora
    if kind in (None, "heur"):
        os.environ.pop("DER_LLM_STRICT", None)
        os.environ.pop("DER_LORA_ADAPTER", None)
        local_lora.reset_default()
        return None, {"llm_backend": kind or "none", "real_llm": False}
    base, adapter = QWEN if kind == "qwen" else LLAMA
    os.environ["DER_LLM_STRICT"] = "1"
    llm = local_lora.configure_default(base, adapter, max_new_tokens=MAX_TOK)
    if not llm._try_load():
        raise RuntimeError(f"FATAL: {kind} backend unavailable: {llm._load_fail_reason}")
    return llm, {"llm_backend": f"{kind}_lora_real", "real_llm": True,
                 "llm_base_model": base, "llm_adapter": adapter,
                 "llm_adapter_sha": llm.adapter_sha(),
                 "llm_model_config_hash": llm.model_config_hash(),
                 "llm_substrate": llm._substrate, "llm_strict": True}


def _make(arm: str, mode: str, name: str, cfg: dict):
    spec = ARMS[arm]
    if spec["kind"] == "deterministic":
        m = importlib.import_module("code.baselines.deterministic_energy_policy.adapter")
        det = m.DeterministicEnergyPolicy()
    elif spec["kind"] == "oracle":
        m = importlib.import_module("code.baselines.oracle_class_policy.adapter")
        det = m.OracleClassPolicy()
        gt = cfg.get("ground_truth") or {}
        det.configure(attack_class=gt.get("attack_class", "none"),
                      target=gt.get("affected_asset") or "",
                      start_s=float(gt.get("start_s", 0) or 0),
                      end_s=float(gt.get("end_s", 0) or 0))
    elif spec["kind"] == "bare":
        m = importlib.import_module("code.baselines.single_llm.runner")
        det = m.SingleLLM(invoke_interval_s=INVOKE_S)
    else:  # full stack, or single LLM + shield (same detector, ablated)
        from ...Multi_AI_Agent.adapter import DERSecAgentDetector
        det = DERSecAgentDetector(
            ablation="full" if spec["kind"] == "full" else "no_incident_memory",
            k_setting=1, compact_prompt=(spec["llm"] != "heur"),
            llm_invoke_interval_s=(INVOKE_S if spec["llm"] != "heur" else None),
            policy_mode=mode)
    det.name = name
    return det


def _decisions(run_dir: Path) -> dict:
    """Per-incident provenance required by the brief."""
    prop = fin = None
    n_dec = n_model_honoured = n_fallback = n_hitl = n_llm = 0
    cand, veto = [], {}
    for line in (run_dir / "decisions.jsonl").read_text().splitlines():
        dt = json.loads(line).get("decision_trace")
        if not dt:
            continue
        n_dec += 1
        n_llm += len(dt.get("llm_traces") or [])
        pj = dt.get("projection") or {}
        if pj:
            n_model_honoured += bool(pj.get("model_proposal_honoured"))
            n_fallback += pj.get("action_source") == "deterministic_fallback"
            cand = pj.get("safe_candidate_set", cand)
            veto = pj.get("safety_veto_reason", veto)
        n_hitl += dt.get("coordinator_decision") == "hitl_required"
        if dt.get("llm_traces"):
            prop, fin = dt.get("proposed_action"), dt.get("final_action")
    return {"n_decision_steps": n_dec, "n_llm_calls": n_llm,
            "n_model_proposal_honoured": n_model_honoured,
            "n_deterministic_fallback": n_fallback, "n_hitl": n_hitl,
            "last_proposed_action": prop, "last_executed_action": fin,
            "safe_candidate_set": ";".join(cand),
            "safety_veto_reason": json.dumps(veto)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="D0,H0,ORC,M1-Q")
    ap.add_argument("--modes", default="legacy_class_override,safety_projection")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1,
                    help="genuine stochastic episodes per configuration (E2); "
                         "1 keeps the deterministic trajectory")
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST)
    if args.limit:
        man = man.head(args.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        llm, meta = _setup_llm(spec["llm"])
        # Policy mode only applies to the shielded multi-agent arms.
        modes = (args.modes.split(",") if spec["kind"] in ("full", "shielded_single")
                 else ["n/a"])
        for mode in modes:
            label = f"{arm}" + (f"__{mode}" if mode != "n/a" else "")
            root = OUT / "runs" / label
            for _, r in man.iterrows():
              cfg = yaml.safe_load(Path(r.config_path).read_text())
              for ep in range(args.episodes):
                # episode_seed=None reproduces the published deterministic
                # trajectory; >=0 draws a genuine exogenous episode (E2).
                ep_seed = None if args.episodes == 1 else ep
                rd = root / cfg["name"] / label / f"seed{ep}"
                if not (rd / "manifest.json").exists():
                    det = _make(arm, mode if mode != "n/a" else "legacy_class_override",
                                label, cfg)
                    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                        ders=cfg["ders"],
                                        base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                                        episode_seed=ep_seed)
                    t0 = time.time()
                    run_scenario(r.config_path, det, ep, out_root=str(root), feeder=feeder,
                                 extra_manifest={**meta, "revision_tag": TAG, "arm": arm,
                                                 "policy_mode": mode, "episode": ep,
                                                 "episode_seed": ep_seed,
                                                 "stochastic_episode": ep_seed is not None,
                                                 "scenario_config_hash": r.configuration_hash})
                    print(f"[done] {label} {cfg['name']} ep{ep} {time.time()-t0:.1f}s", flush=True)
                ph = sweep(rd, np.linspace(0, 1, 11))
                a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]
                rec = {"arm": arm, "policy_mode": mode, "label": label,
                       "episode": ep, "scenario": r.scenario_id,
                       "configuration_hash": r.configuration_hash,
                       "is_control": bool(r.is_calibration_control),
                       "real_llm": meta.get("real_llm", False),
                       "llm_adapter_sha": meta.get("llm_adapter_sha"),
                       **{m: float(a[m]) for m in ("ens_kwh", "curt_kwh",
                                                   "voltage_frac", "ramp_violations")},
                       **_decisions(rd)}
                rows.append(rec)
    df = pd.DataFrame(rows)
    p = OUT / "e1_summary.csv"
    if p.exists():
        df = (pd.concat([pd.read_csv(p), df], ignore_index=True)
              .drop_duplicates(subset=["label", "scenario", "episode"], keep="last"))
    df.to_csv(p, index=False)
    print(f"E1 summary -> {p} ({len(df)} rows)")


if __name__ == "__main__":
    main()
