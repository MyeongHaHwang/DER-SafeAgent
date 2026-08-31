"""P1-A: real QLoRA -> real OpenDSS power flow on a pre-registered subset.

Completes the causal chain REAL QLoRA -> REAL runtime assurance -> REAL
executed action -> REAL OpenDSS consequence. The 12-configuration subset is
drawn from the validated 20-configuration sweep library and FROZEN (with
SHA-256 of each generated scenario file) before any model run.

Systems:
  OD0   deterministic fast path (no LLM)
  OQ1   DER-SafeAgent, real Qwen QLoRA K=1, class-override shield
  OL1   DER-SafeAgent, real Llama QLoRA K=1, class-override shield
  OQP   DER-SafeAgent, real Qwen QLoRA K=1, corrected safety projection
  OLP   DER-SafeAgent, real Llama QLoRA K=1, corrected safety projection

Strict serving integrity: every LLM arm aborts if the intended backbone or
adapter is not loaded, records adapter SHA + prompt hash per call, and any
fallback in a decision trace aborts the run at validation time.

Environment note (documented, not hidden): the OpenDSS harness telemetry is
noiseless, so the duplicate-payload staleness signature is degenerate there
and the Evidence Gate is effectively open; the gate's discriminative
evaluation is the stochastic StubFeeder test (P0-B). LLM arms here therefore
run the full stack WITHOUT the gate wrapper, matching the sweep protocol.

Run: python3 -m code.evaluation.final_safeagent.run_p1a_opendss --freeze
     python3 -m code.evaluation.final_safeagent.run_p1a_opendss --systems OD0
     python3 -m code.evaluation.final_safeagent.run_p1a_opendss --systems OQ1,OQP
     python3 -m code.evaluation.final_safeagent.run_p1a_opendss --systems OL1,OLP
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..physical_curves import sweep
from ..final_revision.run_e1 import _setup_llm
from ..opendss_sweep.run_opendss_sweep import (IEEE13_DIR, IEEE34_DIR,
                                               build_configs)
from ...simulation.feeder import OpenDSSFeeder
from ...simulation.harness import run_scenario

TAG = "ijcip_final_safeagent_20260810"
OUT = Path("code/results") / TAG / "p1a_opendss"
CONF = Path("code/configs") / TAG

SUBSET = ["ods13_spoof_low", "ods13_spoof_med", "ods13_spoof_high",
          "ods13_fdi_low", "ods13_fdi_high", "ods13_dos_med",
          "ods13_spoof_med_peak", "ods13_spoof_med_highpen",
          "ods34_spoof_inv840", "ods34_spoof_bess848", "ods34_spoof_fleet",
          "ods13_benign"]

SYSTEMS = {
    "OD0": dict(llm=None, mode=None),
    "OQ1": dict(llm="qwen", mode="legacy_class_override"),
    "OL1": dict(llm="llama", mode="legacy_class_override"),
    "OQP": dict(llm="qwen", mode="safety_projection"),
    "OLP": dict(llm="llama", mode="safety_projection"),
}


def _scenario_files() -> list[dict]:
    configs = {c["id"]: c for c in build_configs()}
    missing = [s for s in SUBSET if s not in configs]
    assert not missing, f"unknown subset ids: {missing}"
    out = []
    for sid in SUBSET:
        c = configs[sid]
        feeder_dir = IEEE13_DIR if c["family"] == "ieee13" else IEEE34_DIR
        scen_dir = OUT / "scenarios" / sid
        scen_dir.mkdir(parents=True, exist_ok=True)
        cfg_yaml = {
            "name": sid, "description": c["why"],
            "feeder": str((feeder_dir / "feeder.dss").resolve()),
            "dt_s": 1.0, "duration_s": 600,
            "monitored_buses": c["buses"], "ders": c["ders"],
            "attacks": c["attacks"],
            "ground_truth": {"attack_class": c["gt_class"],
                             "affected_asset": c["gt_asset"],
                             "start_s": (c["attacks"][0]["start_s"] if c["attacks"] else 0),
                             "end_s": (c["attacks"][0]["end_s"] if c["attacks"] else 0)},
            "opendss_init_commands": c["init_commands"],
        }
        p = scen_dir / "config.yaml"
        p.write_text(yaml.safe_dump(cfg_yaml, sort_keys=False))
        out.append({"id": sid, "cfg": cfg_yaml, "path": p, "spec": c})
    return out


def freeze() -> None:
    CONF.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in _scenario_files():
        c = s["spec"]
        rows.append(dict(configuration_id=s["id"], feeder=c["family"],
                         attack_type=c["gt_class"], magnitude=c["magnitude"],
                         duration=c["duration"], load_level=c["load_level"],
                         der_penetration=c["der_penetration"],
                         location=c["location"],
                         config_sha256=hashlib.sha256(
                             s["path"].read_bytes()).hexdigest(),
                         config_path=str(s["path"].resolve())))
    df = pd.DataFrame(rows)
    df.to_csv(CONF / "opendss_llm_subset.csv", index=False)
    note = dict(frozen_at="2026-08-10",
                rule="subset selected & frozen BEFORE any model execution",
                n=len(rows),
                manifest_sha256=hashlib.sha256(
                    (CONF / "opendss_llm_subset.csv").read_bytes()).hexdigest())
    (CONF / "opendss_llm_subset_freeze.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))


def _mk_detector(name: str):
    spec = SYSTEMS[name]
    if spec["llm"] is None:
        from ...baselines.deterministic_energy_policy.adapter import (
            DeterministicEnergyPolicy)
        det = DeterministicEnergyPolicy()
    else:
        from ...Multi_AI_Agent.adapter import DERSecAgentDetector
        det = DERSecAgentDetector(
            ablation="full", k_setting=1, compact_prompt=True,
            llm_invoke_interval_s=600.0, policy_mode=spec["mode"],
            fallback_estimator=("eh" if spec["mode"] == "safety_projection"
                                else "class_table"))
    det.name = name
    return det


def _decision_summary(rd: Path) -> dict:
    prop = fin = None
    n_llm = n_hitl = n_fallback = 0
    shield_intervened = False
    lat_ms = []
    for line in (rd / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        dt_ = d.get("decision_trace") or {}
        traces = dt_.get("llm_traces") or []
        n_llm += len(traces)
        for tr in traces:
            assert not tr.get("fallback_reason"), \
                f"{rd}: heuristic fallback in an LLM-labelled run"
            lat_ms.append(float(tr.get("latency_ms", 0.0)))
        if traces:
            prop, fin = dt_.get("proposed_action"), dt_.get("final_action")
            if prop and fin and prop != fin:
                shield_intervened = True
            pj = dt_.get("projection") or {}
            if pj and not pj.get("model_proposal_honoured"):
                shield_intervened = True
            n_fallback += (pj.get("action_source") == "deterministic_fallback")
        n_hitl += (dt_.get("coordinator_decision") == "hitl_required")
    return {"last_proposed_action": prop, "last_final_action": fin,
            "shield_intervened": shield_intervened, "n_llm_calls": n_llm,
            "n_hitl": n_hitl, "n_projection_fallback": n_fallback,
            "llm_latency_ms_mean": (float(np.mean(lat_ms)) if lat_ms else None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--systems", default="")
    args = ap.parse_args()
    if args.freeze:
        freeze()
        return

    import opendssdirect  # noqa: F401 — fail loudly if absent

    man = pd.read_csv(CONF / "opendss_llm_subset.csv")
    scen = {s["id"]: s for s in _scenario_files()}
    # freeze integrity: regenerated configs must match the frozen hashes
    for _, r in man.iterrows():
        h = hashlib.sha256(scen[r.configuration_id]["path"].read_bytes()).hexdigest()
        assert h == r.config_sha256, f"config drift: {r.configuration_id}"

    rows = []
    for name in args.systems.split(","):
        spec = SYSTEMS[name]
        llm, meta = _setup_llm(spec["llm"])
        for _, r in man.iterrows():
            s = scen[r.configuration_id]
            cfg = s["cfg"]
            det = _mk_detector(name)
            root = OUT / "runs" / name
            rd = root / cfg["name"] / name / "seed0"
            if not (rd / "manifest.json").exists():
                feeder = OpenDSSFeeder(
                    dss_path=cfg["feeder"],
                    monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                    init_commands=cfg["opendss_init_commands"])
                t0 = time.time()
                run_scenario(str(s["path"]), det, seed=0, out_root=str(root),
                             feeder=feeder,
                             extra_manifest={**meta, "revision_tag": TAG,
                                             "system": name,
                                             "backend": "opendss_real",
                                             "policy_mode": spec["mode"],
                                             "config_sha256": r.config_sha256})
                mp = rd / "manifest.json"
                mm = json.loads(mp.read_text())
                mm["opendss_n_solves"] = feeder.n_solves
                mm["opendss_n_nonconverged"] = feeder.n_nonconverged
                mp.write_text(json.dumps(mm, indent=2))
                print(f"[done] {name} {cfg['name']} {time.time()-t0:.1f}s "
                      f"(nonconv={feeder.n_nonconverged})", flush=True)

            mm = json.loads((rd / "manifest.json").read_text())
            assert mm.get("backend") == "opendss_real"
            if spec["llm"] is not None:
                assert mm.get("real_llm") is True and mm.get("llm_adapter_sha"), \
                    f"{rd}: LLM provenance missing"
            ph = sweep(rd, np.linspace(0, 1, 11))
            a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]
            ts = pd.read_csv(rd / "timeseries.csv")
            vcols = [c for c in ts.columns if c.startswith("v_pu_")]
            gt = cfg["ground_truth"]
            first_action = t_first = None
            in_attack_action = t_in_attack = None
            all_actions = []
            for line in (rd / "decisions.jsonl").read_text().splitlines():
                d = json.loads(line)
                if d["actions"]:
                    if first_action is None:
                        first_action = d["actions"][0]["name"]
                        t_first = float(d["t"])
                    all_actions += [x["name"] for x in d["actions"]]
                    if (in_attack_action is None and gt["start_s"]
                            and float(d["t"]) >= float(gt["start_s"])):
                        in_attack_action = d["actions"][0]["name"]
                        t_in_attack = float(d["t"])
            rows.append({
                "system": name, "configuration_id": r.configuration_id,
                "attack_type": r.attack_type, "feeder": r.feeder,
                "is_control": r.attack_type == "none",
                "real_llm": bool(mm.get("real_llm")),
                "adapter_sha": mm.get("llm_adapter_sha"),
                "selected_action": first_action or "none",
                "t_first_action_s": t_first,
                "in_attack_action": in_attack_action or "none",
                "t_in_attack_action_s": t_in_attack,
                "action_names": ";".join(sorted(set(all_actions))) or "none",
                "mitigation_latency_s": (None if t_in_attack is None
                                          or not gt["start_s"]
                                          else max(0.0, t_in_attack - gt["start_s"])),
                "ens_kwh": float(a["ens_kwh"]), "curt_kwh": float(a["curt_kwh"]),
                "voltage_min_pu": float(ts[vcols].min().min()),
                "voltage_max_pu": float(ts[vcols].max().max()),
                "voltage_violation_frac": float(a["voltage_frac"]),
                "n_nonconverged": int(mm.get("opendss_n_nonconverged", 0)),
                **_decision_summary(rd)})
    df = pd.DataFrame(rows)
    p = OUT / "p1a_raw.csv"
    if p.exists():
        df = (pd.concat([pd.read_csv(p), df], ignore_index=True)
              .drop_duplicates(subset=["system", "configuration_id"], keep="last"))
    df.to_csv(p, index=False)
    print(f"-> {p} ({len(df)} rows)")


if __name__ == "__main__":
    main()
