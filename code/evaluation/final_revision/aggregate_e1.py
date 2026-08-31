"""Aggregate E1 from completed run directories (no model load required).

The E1 driver writes its summary at the end of a process; if a later arm fails
to acquire the GPU the summary is lost even though the runs themselves are on
disk. This script rebuilds the summary purely from run artifacts, so the
aggregation never depends on GPU availability.

Run: python3 -m code.evaluation.final_revision.aggregate_e1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..physical_curves import sweep

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "e1_causal"
MANIFEST = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")


def main() -> None:
    man = pd.read_csv(MANIFEST).set_index("scenario_id")
    rows = []
    for man_p in sorted((OUT / "runs").rglob("manifest.json")):
        rd = man_p.parent
        meta = json.loads(man_p.read_text())
        scen = meta["scenario"]
        label = meta.get("detector", rd.parent.name)
        arm = meta.get("arm", label.split("__")[0])
        mode = meta.get("policy_mode", "n/a")

        ph = sweep(rd, np.linspace(0, 1, 11))
        a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]

        n_dec = n_llm = n_honoured = n_fb = n_hitl = n_caution = 0
        n_real = n_fallback_trace = 0
        prop = fin = None
        diverged = 0
        for line in (rd / "decisions.jsonl").read_text().splitlines():
            dt = json.loads(line).get("decision_trace")
            if not dt:
                continue
            n_dec += 1
            traces = dt.get("llm_traces") or []
            n_llm += len(traces)
            n_real += sum(1 for t in traces if t.get("backend") == "real_lora")
            n_fallback_trace += sum(1 for t in traces
                                    if t.get("fallback_reason") is not None)
            pj = dt.get("projection") or {}
            if pj:
                n_honoured += bool(pj.get("model_proposal_honoured"))
                n_fb += pj.get("action_source") == "deterministic_fallback"
            n_hitl += dt.get("coordinator_decision") == "hitl_required"
            n_caution += bool(dt.get("caution"))
            if traces:
                prop, fin = dt.get("proposed_action"), dt.get("final_action")
                if prop and fin and prop != fin:
                    diverged += 1
        rows.append({
            "arm": arm, "policy_mode": mode, "label": label, "scenario": scen,
            "real_llm": bool(meta.get("real_llm", False)),
            "llm_adapter_sha": meta.get("llm_adapter_sha"),
            "configuration_hash": meta.get("scenario_config_hash"),
            "is_control": bool(man.loc[scen, "is_calibration_control"])
                          if scen in man.index else False,
            "ens_kwh": float(a["ens_kwh"]), "curt_kwh": float(a["curt_kwh"]),
            "voltage_frac": float(a["voltage_frac"]),
            "ramp_violations": float(a["ramp_violations"]),
            "n_decision_steps": n_dec, "n_llm_calls": n_llm,
            "n_real_llm_calls": n_real, "n_fallback_traces": n_fallback_trace,
            "n_model_proposal_honoured": n_honoured,
            "n_deterministic_fallback": n_fb, "n_hitl": n_hitl,
            "n_caution": n_caution, "n_proposed_final_divergent": diverged,
            "last_proposed_action": prop, "last_executed_action": fin,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_summary.csv", index=False)

    # Integrity: no run labelled real_llm may contain a fallback trace.
    bad = df[(df.real_llm) & (df.n_fallback_traces > 0)]
    assert bad.empty, f"FATAL: {len(bad)} real-LLM runs contain fallback traces"

    live = df[~df.is_control]
    agg = (live.groupby(["arm", "policy_mode"])
           .agg(n_cfg=("scenario", "nunique"),
                ens=("ens_kwh", "mean"), curt=("curt_kwh", "mean"),
                real_calls=("n_real_llm_calls", "sum"),
                fallback_traces=("n_fallback_traces", "sum"),
                honoured=("n_model_proposal_honoured", "sum"),
                det_fallback=("n_deterministic_fallback", "sum"),
                divergent=("n_proposed_final_divergent", "sum"))
           .round(3))
    agg.to_csv(OUT / "e1_by_arm.csv")
    print(agg.to_string())
    print(f"\nrows -> {OUT/'e1_summary.csv'}  ({len(df)})")


if __name__ == "__main__":
    main()
