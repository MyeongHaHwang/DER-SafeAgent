"""P0 forensic audit (B, C, D) over the real holdout results, with assertions.

P0-A (gate determinism) is in p0a_gate_forensics.py. This module resolves:
  P0-B  LLM call-count unit reconciliation
  P0-C  HITL escalation asymmetry (backbone- and mode-dependent)
  P0-D  bareQ/bareL identical physical outcomes (lineage: genuine, not reuse)

Writes code/results/ijcip_final_v3/p0_forensics/p0bcd_result.json and asserts
the P0-D lineage invariant (identical physical trajectory across different
backbones is allowed ONLY when the executed-action sequences are identical).

Run: python3 -m code.evaluation.ijcip_final_v3.p0_forensics
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import pandas as pd

RES = Path("code/results/ijcip_final_v3")
RUNS = RES / "holdout_e2e" / "runs"
OUT = RES / "p0_forensics"


def _exec_seq(arm, scen, seed):
    f = RUNS / arm / scen / arm / f"seed{seed}" / "decisions.jsonl"
    seq = []
    for ln in f.read_text().splitlines():
        d = json.loads(ln)
        for a in (d.get("actions") or []):
            seq.append((d["t"], a.get("kind") if isinstance(a, dict) else a))
    return seq


def _ts_hash(arm, scen, seed):
    f = RUNS / arm / scen / arm / f"seed{seed}" / "timeseries.csv"
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RES / "holdout_e2e" / "holdout_e2e_raw.csv")
    atk = df[df.kind == "attack"]
    ben = df[df.kind == "benign"]
    out = {}

    # ---- P0-B: call-count unit ----
    b = {}
    for arm in ["Q1", "L1", "QPROJ", "LPROJ", "bareQ", "bareL"]:
        a = atk[atk.arm == arm]
        b[arm] = {
            "table8_calls_sum_of_perconfig_means": round(a.groupby("scenario").n_llm_calls.mean().sum(), 1),
            "raw_sum_over_99_attack_rows": int(a.n_llm_calls.sum()),
            "mean_calls_per_attack_episode": round(a.n_llm_calls.mean(), 2),
            "attack_configs_with_ge1_call": int((a.groupby("scenario").n_llm_calls.mean() > 0).sum()),
        }
    out["P0B_call_counts"] = {
        "unit": ("Table 8 LLM-calls = sum over 33 attack configs of the per-config "
                 "MEAN calls (episodes averaged). A sustained incident re-consults "
                 "across ticks, so calls/incident > 1; 49x3 does not bound calls."),
        "per_arm": b,
    }

    # ---- P0-C: HITL asymmetry ----
    c = {}
    for arm in ["QPROJ", "LPROJ", "Q1", "L1"]:
        a = atk[atk.arm == arm]
        bn = ben[ben.arm == arm]
        esc = a.groupby("scenario").n_hitl.mean()
        c[arm] = {
            "total_hitl": int(a.n_hitl.sum()),
            "esc_per_attack_config_mean": round(esc.mean(), 3),
            "attack_configs_escalated": int((esc > 0).sum()),
            "benign_configs_escalated": int((bn.groupby("scenario").n_hitl.mean() > 0).sum()),
        }
    # tie to irreversible-proposal rates in the adversarial suite
    irr = {}
    for be in ["qwen", "llama"]:
        p = RES / "adversarial" / f"adversarial_raw_{be}.csv"
        if p.exists():
            ad = pd.read_csv(p)
            bl = ad[ad.arm == "bare_llm"]
            irr[be] = {"bare_isolate_proposed": int((bl.proposed == "isolate_inverter").sum()),
                       "n": int(len(bl))}
    out["P0C_hitl_asymmetry"] = {
        "root_cause": ("Escalation is triggered by irreversible (isolate) proposals under "
                       "projection mode. Qwen proposes isolate far more than Llama, and only "
                       "projection arms route irreversible proposals to HITL; class-override "
                       "arms (Q1/L1) overwrite the proposal, so they never escalate. Burden is "
                       "backbone- AND mode-dependent; no human study, so no acceptability claim."),
        "per_arm": c,
        "adversarial_irreversible_proposal": irr,
    }

    # ---- P0-D: bareQ/bareL lineage ----
    scens = sorted(atk[atk.arm == "bareQ"].scenario.unique())
    identical_ts = same_exec = diff_adapter = 0
    violations = []
    for s in scens:
        for seed in (0, 1, 2):
            hq, hl = _ts_hash("bareQ", s, seed), _ts_hash("bareL", s, seed)
            eq, el = _exec_seq("bareQ", s, seed), _exec_seq("bareL", s, seed)
            if hq is not None and hq == hl:
                identical_ts += 1
                # ASSERTION: identical physical trajectory across different
                # backbones is legitimate ONLY if executed actions are identical
                if eq == el:
                    same_exec += 1
                else:
                    violations.append({"scenario": s, "seed": seed,
                                       "bareQ_exec": eq, "bareL_exec": el})
    adap = {"bareQ": sorted(atk[atk.arm == "bareQ"].adapter_sha.dropna().unique().tolist()),
            "bareL": sorted(atk[atk.arm == "bareL"].adapter_sha.dropna().unique().tolist())}
    out["P0D_bare_lineage"] = {
        "finding": ("bareQ and bareL are byte-identical physically because both execute ZERO "
                    "actions: the ungated arm polls the LLM on a fixed 600 s cadence that "
                    "misses the short attack window, so the model returns no_op and the feeder "
                    "evolves as the unmitigated attack, independent of backbone. Genuine, not "
                    "reuse: adapters/prompts/outputs differ and are logged."),
        "distinct_adapters": adap,
        "episodes_with_identical_timeseries": identical_ts,
        "of_which_identical_executed_actions": same_exec,
        "LINEAGE_ASSERTION_violations": len(violations),
        "PASS": len(violations) == 0,
        "caveat": ("Holdout bare arm is a no-effective-response baseline (coarse polling), NOT "
                   "a demonstration of unsafe execution; the latter is the adversarial suite."),
    }
    assert len(violations) == 0, f"P0-D lineage violation: {violations[:3]}"

    (OUT / "p0bcd_result.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print("\nP0-D lineage assertion PASS:", len(violations) == 0,
          f"({same_exec}/{identical_ts} identical-timeseries episodes have identical executed actions)")


if __name__ == "__main__":
    main()
