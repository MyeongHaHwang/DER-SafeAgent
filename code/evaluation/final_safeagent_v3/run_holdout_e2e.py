"""V3 Phase 2 — end-to-end evaluation on the frozen holdout (runtime-safe).

Runs the frozen 49-config holdout x 3 episodes through the de-oracled
(runtime_safe) architecture. Deterministic/oracle arms (D0, OPROJ) need no
GPU; the LLM arms (bareQ/bareL, Q1/L1, QPROJ/LPROJ) load a real adapter under
strict serving and abort on any fallback.

Arms use the Evidence Gate (runtime_safe) and emit_seq feeder so the observable
replay/DoS/FDI signatures are available. Ground truth (attack class/window) is
used only by the offline metric computation, never by the runtime path.

Run:
  python3 -m code.evaluation.final_safeagent_v3.run_holdout_e2e --arms D0,OPROJ
  python3 -m code.evaluation.final_safeagent_v3.run_holdout_e2e --arms Q1,QPROJ  # GPU
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
from code.simulation.portable_paths import resolve_config_path as _resolve

CONF = Path("code/configs/ijcip_final_v3")
OUT = Path("code/results/ijcip_final_v3/holdout_e2e")
EPISODE_SEEDS = [1301, 1302, 1303]

ARMS = {
    "D0":    dict(llm=None, mode=None, kind="deterministic"),
    "OPROJ": dict(llm=None, mode="safety_projection", kind="oracle_projection"),
    "bareQ": dict(llm="qwen", mode=None, kind="bare"),
    "bareL": dict(llm="llama", mode=None, kind="bare"),
    "Q1":    dict(llm="qwen", mode="legacy_class_override", kind="full"),
    "L1":    dict(llm="llama", mode="legacy_class_override", kind="full"),
    "QPROJ": dict(llm="qwen", mode="safety_projection", kind="full"),
    "LPROJ": dict(llm="llama", mode="safety_projection", kind="full"),
}


def _feeder(cfg, seed):
    sp = cfg.get("stub_params") or {}
    return StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                      base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                      episode_seed=sp.get("episode_seed", seed),
                      load_sigma=float(sp.get("load_sigma", 0.08)),
                      pv_cloud_rate=float(sp.get("pv_cloud_rate", 0.004)),
                      meas_noise_kw=float(sp.get("meas_noise_kw", 0.8)),
                      emit_seq=True)


def _mk(arm, cfg, params):
    spec = ARMS[arm]
    from .run_gate_robustness import _fdi_mag_configs  # noqa: keep import light
    if spec["kind"] == "deterministic":
        from ...baselines.deterministic_energy_policy.adapter import (
            DeterministicEnergyPolicy)
        det = DeterministicEnergyPolicy(runtime_safe=True)
    elif spec["kind"] == "oracle_projection":
        from ..final_safeagent.run_p1b_projection import OracleProjectionDetector
        det = OracleProjectionDetector()
        det.configure(cfg.get("ground_truth") or {})
    elif spec["kind"] == "bare":
        from ...baselines.single_llm.runner import SingleLLM
        det = SingleLLM(invoke_interval_s=600.0)
    else:
        from ...Multi_AI_Agent.adapter import DERSecAgentDetector
        det = DERSecAgentDetector(
            ablation="full", k_setting=1, compact_prompt=True,
            llm_invoke_interval_s=600.0, policy_mode=spec["mode"],
            runtime_safe=True,
            fallback_estimator=("eh" if spec["mode"] == "safety_projection"
                                else "class_table"))
    # bare arm has no gate (registry-only control); all others gated
    if spec["kind"] == "bare":
        det.name = arm
        return det
    det = GatedDetector(inner=det, params=params, runtime_safe=True)
    det.name = arm
    return det


def _metrics(rd, cfg):
    ph = sweep(rd, np.linspace(0, 1, 11))
    a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]
    gt = cfg.get("ground_truth") or {}
    s0, s1 = float(gt.get("start_s") or 0), float(gt.get("end_s") or 0)
    n_act = n_hitl = n_llm = 0
    first = None
    covered = False
    fallbacks = 0
    for line in (rd / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        t = float(d["t"])
        if d["actions"]:
            n_act += len(d["actions"])
            if first is None:
                first = t
        if s0 and s0 <= t <= s1 + 60 and \
                d["detection"].get("attack_class", "none") != "none":
            covered = True
        dt_ = d.get("decision_trace") or {}
        for tr in (dt_.get("llm_traces") or []):
            n_llm += 1
            if tr.get("fallback_reason"):
                fallbacks += 1
        n_hitl += (dt_.get("coordinator_decision") == "hitl_required")
    return {"ens_kwh": float(a["ens_kwh"]), "curt_kwh": float(a["curt_kwh"]),
            "voltage_frac": float(a["voltage_frac"]), "n_actions": n_act,
            "t_first_s": first, "gate_open_covered": covered,
            "n_hitl": n_hitl, "n_llm_calls": n_llm, "n_fallbacks": fallbacks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # runtime-safe gate params (frozen in Phase 1)
    fz = json.loads((CONF / "evidence_gate_v3_frozen.json").read_text())
    params = GateParams(p_hard=fz["p_hard"], p_soft=fz["p_soft"],
                        z_soft=fz["z_soft"])
    man = pd.read_csv(CONF / "holdout_v3_manifest.csv")

    rows = []
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        _, meta = _setup_llm(spec["llm"])
        for _, r in man.iterrows():
            cfg = yaml.safe_load(_resolve(r.config_path).read_text())
            for ep, seed in enumerate(EPISODE_SEEDS):
                root = OUT / "runs" / arm
                rd = root / cfg["name"] / arm / f"seed{ep}"
                if not (rd / "manifest.json").exists():
                    det = _mk(arm, cfg, params)
                    t0 = time.time()
                    run_scenario(str(_resolve(r.config_path)), det, seed=ep, out_root=str(root),
                                 feeder=_feeder(cfg, seed),
                                 extra_manifest={**meta, "tag": "ijcip_final_v3",
                                                 "arm": arm, "runtime_safe": True,
                                                 "policy_mode": spec["mode"],
                                                 "episode_seed": seed,
                                                 "config_sha256": r.config_sha256})
                    if spec["llm"]:
                        print(f"[done] {arm} {cfg['name']} ep{ep} "
                              f"{time.time()-t0:.1f}s", flush=True)
                m = _metrics(rd, cfg)
                assert m["n_fallbacks"] == 0, f"FALLBACK in {rd}"
                rows.append({"arm": arm, "scenario": r.scenario_id,
                             "kind": r.kind, "attack_type": r.attack_type,
                             "episode": ep, "real_llm": bool(meta.get("real_llm")),
                             "adapter_sha": meta.get("llm_adapter_sha"), **m})
    df = pd.DataFrame(rows)
    p = OUT / "holdout_e2e_raw.csv"
    if p.exists():
        df = (pd.concat([pd.read_csv(p), df], ignore_index=True)
              .drop_duplicates(subset=["arm", "scenario", "episode"], keep="last"))
    df.to_csv(p, index=False)
    print(f"-> {p} ({len(df)} rows)")


if __name__ == "__main__":
    main()
