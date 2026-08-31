"""P1-B: safety-projection re-evaluation under the FINAL architecture.

Arms (all behind the frozen Evidence Gate — the final-architecture posture):
  D0     deterministic fast path
  Q1/L1  DER-SafeAgent, real Qwen/Llama QLoRA K=1, class-override shield
  QPROJ/LPROJ  same backbones under the corrected safety projection
               (irreversible primitives never auto-executed)
  OPROJ  ground-truth class through the same projection logic (no model)

Frozen 25-configuration library x 3 genuine stochastic episodes
(episode seeds 0/1/2 - exogenous randomness only, never deterministic seed
replication). Strict serving for all LLM arms.

Run: python3 -m code.evaluation.final_safeagent.run_p1b_projection --arms D0,OPROJ
     python3 -m code.evaluation.final_safeagent.run_p1b_projection --arms Q1,QPROJ
     python3 -m code.evaluation.final_safeagent.run_p1b_projection --arms L1,LPROJ
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
from ...simulation.types import Action, Detection
from code.simulation.portable_paths import resolve_config_path as _resolve

TAG = "ijcip_final_safeagent_20260810"
OUT = Path("code/results") / TAG / "p1b_projection"
CONF = Path("code/configs") / TAG
MANIFEST = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")
EPISODES = 3

ARMS = {
    "D0":    dict(llm=None, mode=None, kind="deterministic"),
    "OPROJ": dict(llm=None, mode="safety_projection", kind="oracle_projection"),
    "Q1":    dict(llm="qwen", mode="legacy_class_override", kind="full"),
    "L1":    dict(llm="llama", mode="legacy_class_override", kind="full"),
    "QPROJ": dict(llm="qwen", mode="safety_projection", kind="full"),
    "LPROJ": dict(llm="llama", mode="safety_projection", kind="full"),
}


class OracleProjectionDetector:
    """Ground-truth class fed through the SAME projection logic (no model).

    proposed_action=None => the projection's deterministic fallback chooses
    within the class-restricted safe candidate set. This is the upper-bound
    reference for what the projection policy can achieve with a perfect
    classifier.
    """
    name = "OPROJ"

    def __init__(self) -> None:
        self._gt = {"attack_class": "none", "affected_asset": "",
                    "start_s": 0.0, "end_s": 0.0}
        self._telemetry: list = []
        self._events: list = []
        self._last_decision = None

    def configure(self, gt: dict) -> None:
        self._gt = {**self._gt, **{k: v for k, v in gt.items() if v is not None}}

    def step(self, t, telemetry, events):
        from ...Multi_AI_Agent.safety_projection import project
        from ...Multi_AI_Agent.telemetry_features import extract

        self._telemetry.append({
            "t": telemetry.t, "freq_hz": telemetry.freq_hz,
            "v_pu": telemetry.bus_voltages_pu, "der_p_kw": telemetry.der_p_kw,
            "der_p_avail": telemetry.der_p_avail_kw,
            "load_demand_kw": telemetry.load_demand_kw,
            "load_served_kw": telemetry.load_served_kw})
        self._telemetry = self._telemetry[-60:]
        self._events += [{"t": e.t, "source": e.source, "kind": e.kind,
                          "payload": e.payload, "tampered": e.tampered}
                         for e in events]
        self._events = self._events[-60:]
        active = (float(self._gt.get("start_s") or 0)
                  <= t <= float(self._gt.get("end_s") or 0))
        cls = self._gt["attack_class"] if active else "none"
        if cls == "none":
            return Detection(asset=None, attack_class="none", confidence=0.0), []
        fv = extract(telemetry_window=self._telemetry, event_window=self._events)
        proj = project(attack_class=cls, proposed_action=None, confidence=1.0,
                       severity=max(fv.severity_score, 0.31), cautioned=False,
                       parse_ok=False)
        target = self._gt.get("affected_asset") or fv.dominant_asset or ""
        # Same projection logic as the final architecture: EH ranks the safe
        # candidate set on deterministic fallback.
        if proj.source == "deterministic_fallback" and len(proj.safe_candidate_set) > 1:
            try:
                from ...Multi_AI_Agent.horizon_estimator import (
                    HorizonCalibration, class_family, estimate_eh, select)
                from pathlib import Path as _P
                if not hasattr(self, "_calib"):
                    self._calib = HorizonCalibration.load(_P(
                        "code/configs/ijcip_final_safeagent_20260810/"
                        "eh_duration_prior.json"))
                tw = self._telemetry
                load_kw = float(tw[-1]["load_demand_kw"]) if tw else None
                elapsed = max(0.0, t - float(self._gt.get("start_s") or t))
                choice = select(
                    estimate_eh(fv, target, elapsed, self._calib,
                                load_demand_kw=load_kw),
                    family=class_family(fv),
                    severity=max(fv.severity_score, 0.31)).action
                if choice in proj.safe_candidate_set:
                    proj.executed_action = choice
            except Exception:
                pass
        self._last_decision = {
            "t": t, "attack_class": cls, "final_action": proj.executed_action,
            "final_action_target": target,
            "coordinator_decision": ("hitl_required" if proj.hitl_required
                                     else "auto"),
            "projection": proj.to_dict()}
        det = Detection(asset=target or None, attack_class=cls, confidence=1.0,
                        rationale="oracle+projection")
        if proj.executed_action == "no_op" or proj.hitl_required:
            return det, []
        return det, [Action(name=proj.executed_action, target=target)]


def _mk(arm: str, cfg: dict, params: GateParams):
    spec = ARMS[arm]
    if spec["kind"] == "deterministic":
        from ...baselines.deterministic_energy_policy.adapter import (
            DeterministicEnergyPolicy)
        det = DeterministicEnergyPolicy()
    elif spec["kind"] == "oracle_projection":
        det = OracleProjectionDetector()
        det.configure(cfg.get("ground_truth") or {})
    else:
        from ...Multi_AI_Agent.adapter import DERSecAgentDetector
        det = DERSecAgentDetector(
            ablation="full", k_setting=1, compact_prompt=True,
            llm_invoke_interval_s=600.0, policy_mode=spec["mode"],
            # final architecture: EH ranks the safe set when the projection
            # falls back deterministically (P0-A CASE A)
            fallback_estimator=("eh" if spec["mode"] == "safety_projection"
                                else "class_table"))
    det = GatedDetector(inner=det, params=params)
    det.name = arm
    return det


def _decisions(rd: Path) -> dict:
    n_llm = n_hitl = n_honoured = n_fallback = n_override = 0
    prop = fin = None
    for line in (rd / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        dt_ = d.get("decision_trace") or {}
        traces = dt_.get("llm_traces") or []
        n_llm += len(traces)
        for tr in traces:
            assert not tr.get("fallback_reason"), f"{rd}: fallback detected"
        pj = dt_.get("projection") or {}
        if pj:
            n_honoured += bool(pj.get("model_proposal_honoured"))
            n_fallback += (pj.get("action_source") == "deterministic_fallback")
            n_override += (pj.get("action_source") == "caution_gate")
        n_hitl += (dt_.get("coordinator_decision") == "hitl_required")
        if traces:
            prop, fin = dt_.get("proposed_action"), dt_.get("final_action")
    return {"n_llm_calls": n_llm, "n_hitl": n_hitl,
            "n_model_honoured": n_honoured, "n_det_fallback": n_fallback,
            "n_safety_override": n_override,
            "last_proposed": prop, "last_final": fin}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = GateParams.load(CONF / "evidence_gate_frozen.json")
    man = pd.read_csv(MANIFEST)

    rows = []
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        _, meta = _setup_llm(spec["llm"])
        for _, r in man.iterrows():
            cfg = yaml.safe_load(_resolve(r.config_path).read_text())
            for ep in range(EPISODES):
                root = OUT / "runs" / arm
                rd = root / cfg["name"] / arm / f"seed{ep}"
                if not (rd / "manifest.json").exists():
                    det = _mk(arm, cfg, params)
                    feeder = StubFeeder(
                        monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                        base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                        episode_seed=ep)
                    t0 = time.time()
                    run_scenario(str(_resolve(r.config_path)), det, seed=ep, out_root=str(root),
                                 feeder=feeder,
                                 extra_manifest={**meta, "revision_tag": TAG,
                                                 "arm": arm,
                                                 "policy_mode": spec["mode"],
                                                 "episode_seed": ep,
                                                 "stochastic_episode": True,
                                                 "evidence_gate": True,
                                                 "scenario_config_hash":
                                                     r.configuration_hash})
                    print(f"[done] {arm} {cfg['name']} ep{ep} "
                          f"{time.time()-t0:.1f}s", flush=True)
                ph = sweep(rd, np.linspace(0, 1, 11))
                a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]
                rows.append({"arm": arm, "scenario": r.scenario_id,
                             "configuration_hash": r.configuration_hash,
                             "episode": ep,
                             "is_control": bool(r.is_calibration_control),
                             "real_llm": bool(meta.get("real_llm")),
                             "ens_kwh": float(a["ens_kwh"]),
                             "curt_kwh": float(a["curt_kwh"]),
                             "voltage_frac": float(a["voltage_frac"]),
                             **_decisions(rd)})
    df = pd.DataFrame(rows)
    p = OUT / "p1b_raw.csv"
    if p.exists():
        df = (pd.concat([pd.read_csv(p), df], ignore_index=True)
              .drop_duplicates(subset=["arm", "scenario", "episode"], keep="last"))
    df.to_csv(p, index=False)
    print(f"-> {p} ({len(df)} rows)")


if __name__ == "__main__":
    main()
