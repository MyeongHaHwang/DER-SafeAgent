"""P0-B summary: gate test-set metrics table from gate_test_raw.csv.

Run: python3 -m code.evaluation.final_safeagent.summarize_p0b
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "ijcip_final_safeagent_20260810"
OUT = Path("code/results") / TAG / "p0b_gate"


def main() -> None:
    df = pd.read_csv(OUT / "gate_test_raw.csv")
    null = df[df.system == "NOACT"].set_index("scenario")
    rows = []
    for sysname in sorted(set(df.system) - {"NOACT"}):
        d = df[df.system == sysname]
        ben = d[d.condition_kind.str.startswith("benign")].set_index("scenario")
        atk = d[d.condition_kind == "attack"]
        unnec = (ben.curt_kwh - null.curt_kwh.reindex(ben.index)).mean()
        n_ben = len(ben)
        n_atk = len(atk)
        fa = float((ben.n_actions > 0).mean())
        rows.append({
            "system": sysname,
            "n_benign": n_ben, "n_attack": n_atk,
            "benign_false_action_rate": fa,
            "benign_FA_normal": float((ben[ben.condition_kind == "benign_normal"]
                                       .n_actions > 0).mean()),
            "benign_FA_suspicious": float((ben[ben.condition_kind == "benign_suspicious"]
                                           .n_actions > 0).mean()),
            "specificity": 1.0 - fa,
            "unnecessary_curt_kwh": float(unnec),
            "benign_llm_calls": int(ben.n_llm_calls.sum()),
            "attack_coverage": float(atk.attack_covered.mean()),
            "attack_miss_rate": float(1.0 - atk.attack_covered.mean()),
            "attack_mitigated": float((atk.n_actions > 0).mean()),
            "attack_ens_kwh_mean": float(atk.ens_kwh.mean()),
            "attack_t_first_mean_s": float(atk.t_first_action_s.mean()),
            "attack_llm_calls": int(atk.n_llm_calls.sum()),
            "hitl_total": int(d.n_hitl.sum()),
            "step_p95_ms_mean": float(d.step_p95_ms.mean()),
        })
    out = pd.DataFrame(rows)
    out.round(4).to_csv(OUT / "p0b_summary.csv", index=False)
    print(out.round(3).to_string(index=False))

    # selective-coverage detail: which benign scenarios still act under G1
    for sysname in [s for s in out.system if s.endswith("G1")]:
        ben = df[(df.system == sysname)
                 & df.condition_kind.str.startswith("benign")]
        acts = ben[ben.n_actions > 0]
        print(f"\n{sysname} residual benign actions: "
              f"{acts[['scenario', 'action_names']].to_dict('records')}")


if __name__ == "__main__":
    main()
