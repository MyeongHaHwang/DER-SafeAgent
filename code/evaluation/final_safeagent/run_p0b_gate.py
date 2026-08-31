"""P0-B: closed-loop Evidence-Gate evaluation on the FROZEN 32-config test set.

Systems (gate params from the frozen calibration, never re-tuned here):
  D0-G0    deterministic energy policy, no gate
  D0-G1    deterministic energy policy behind the Evidence Gate
  Q1-G0    full DER-SafeAgent stack, real Qwen QLoRA K=1, no gate
  Q1-G1    same, behind the Evidence Gate
  NULL     never acts (paired baseline for unnecessary-curtailment accounting)

Strict serving integrity for Q1 arms: DER_LLM_STRICT=1, adapter SHA per call,
abort on any fallback.

Run: python3 -m code.evaluation.final_safeagent.run_p0b_gate --systems D0,NULL
     python3 -m code.evaluation.final_safeagent.run_p0b_gate --systems Q1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..physical_curves import sweep
from ..final_revision.run_e1 import _setup_llm
from ...Multi_AI_Agent.evidence_gate import GateParams, GatedDetector
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario
from ...simulation.types import Detection

TAG = "ijcip_final_safeagent_20260810"
OUT = Path("code/results") / TAG / "p0b_gate"
CONF = Path("code/configs") / TAG


class NullDetector:
    name = "null_detector"

    def step(self, t, telemetry, events):
        return Detection(asset=None, attack_class="none", confidence=0.0), []


def _feeder(cfg: dict) -> StubFeeder:
    sp = cfg.get("stub_params") or {}
    return StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                      base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                      episode_seed=sp.get("episode_seed"),
                      load_sigma=float(sp.get("load_sigma", 0.08)),
                      pv_cloud_rate=float(sp.get("pv_cloud_rate", 0.004)),
                      meas_noise_kw=float(sp.get("meas_noise_kw", 0.8)))


def _mk(system: str, gated: bool, params: GateParams):
    if system == "NULL":
        det = NullDetector()
    elif system == "D0":
        from ...baselines.deterministic_energy_policy.adapter import (
            DeterministicEnergyPolicy)
        det = DeterministicEnergyPolicy()
    elif system == "Q1":
        from ...Multi_AI_Agent.adapter import DERSecAgentDetector
        det = DERSecAgentDetector(ablation="full", k_setting=1,
                                  compact_prompt=True,
                                  llm_invoke_interval_s=600.0,
                                  policy_mode="legacy_class_override")
    else:  # QP - final architecture: corrected projection + EH fallback
        from ...Multi_AI_Agent.adapter import DERSecAgentDetector
        det = DERSecAgentDetector(ablation="full", k_setting=1,
                                  compact_prompt=True,
                                  llm_invoke_interval_s=600.0,
                                  policy_mode="safety_projection",
                                  fallback_estimator="eh")
    if gated:
        det = GatedDetector(inner=det, params=params)
    label = f"{system}-{'G1' if gated else 'G0'}" if system != "NULL" else "NOACT"
    det.name = label
    return det


def _run_metrics(rd: Path, cfg: dict) -> dict:
    ph = sweep(rd, np.linspace(0, 1, 11))
    a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]
    gt = cfg.get("ground_truth") or {}
    s0, s1 = float(gt.get("start_s") or 0), float(gt.get("end_s") or 0)
    n_act = 0
    t_first = None
    actions = []
    covered = False
    n_hitl = n_llm = 0
    for line in (rd / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        t = float(d["t"])
        if d["actions"]:
            n_act += len(d["actions"])
            actions += [x["name"] for x in d["actions"]]
            if t_first is None:
                t_first = t
        if s0 and s0 <= t <= s1 + 60.0 and \
                d["detection"].get("attack_class", "none") != "none":
            covered = True
        dt_ = d.get("decision_trace") or {}
        n_hitl += (dt_.get("coordinator_decision") == "hitl_required")
        n_llm += len(dt_.get("llm_traces") or [])
    rt = json.loads((rd / "runtime.json").read_text()) \
        if (rd / "runtime.json").exists() else {}
    return {"ens_kwh": float(a["ens_kwh"]), "curt_kwh": float(a["curt_kwh"]),
            "n_actions": n_act, "t_first_action_s": t_first,
            "action_names": ";".join(sorted(set(actions))),
            "attack_covered": covered, "n_hitl": n_hitl, "n_llm_calls": n_llm,
            "step_p95_ms": rt.get("p95_ms"), "step_mean_ms": rt.get("mean_ms")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="D0,NULL")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = GateParams.load(CONF / "evidence_gate_frozen.json")
    man = pd.read_csv(CONF / "evidence_gate_test.csv")

    rows = []
    for system in args.systems.split(","):
        llm_meta = {}
        if system in ("Q1", "QP"):
            _, llm_meta = _setup_llm("qwen")
        variants = [False] if system == "NULL" else [False, True]
        for gated in variants:
            for _, r in man.iterrows():
                cfg = yaml.safe_load(Path(r.config_path).read_text())
                det = _mk(system, gated, params)
                root = OUT / "runs" / det.name
                rd = root / cfg["name"] / det.name / "seed0"
                if not (rd / "manifest.json").exists():
                    t0 = time.time()
                    run_scenario(r.config_path, det, seed=0, out_root=str(root),
                                 feeder=_feeder(cfg),
                                 extra_manifest={**llm_meta, "revision_tag": TAG,
                                                 "system": system,
                                                 "evidence_gate": gated,
                                                 "gate_params": vars(params) if gated else None,
                                                 "condition_kind": r.condition_kind,
                                                 "config_sha256": r.config_sha256})
                    print(f"[done] {det.name} {cfg['name']} "
                          f"{time.time()-t0:.1f}s", flush=True)
                m = _run_metrics(rd, cfg)
                gate_stats = {}
                if gated and hasattr(det, "n_suppressed_steps"):
                    gate_stats = {"gate_suppressed_steps": det.n_suppressed_steps,
                                  "gate_open_steps": det.n_open_steps}
                rows.append({"system": det.name, "scenario": r.scenario_id,
                             "condition_kind": r.condition_kind,
                             "attack_type": r.attack_type,
                             "config_sha256": r.config_sha256,
                             "real_llm": bool(llm_meta.get("real_llm")),
                             **m, **gate_stats})
    df = pd.DataFrame(rows)
    p = OUT / "gate_test_raw.csv"
    if p.exists():
        df = (pd.concat([pd.read_csv(p), df], ignore_index=True)
              .drop_duplicates(subset=["system", "scenario"], keep="last"))
    df.to_csv(p, index=False)
    print(f"-> {p} ({len(df)} rows)")


if __name__ == "__main__":
    main()
