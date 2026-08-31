"""Runtime-safe OpenDSS physical check (IJCIP §5.2).

Runs the deterministic (D0) and oracle-projection (OPROJ) arms through REAL
OpenDSS IEEE-13/34 power-flow solves on the pre-registered subset, under the
observable-only (runtime_safe) runtime. This is the real-power-flow validation
that qualifies the StubFeeder holdout (a fast, stochastic diagnostic
environment). Non-LLM arms only (CPU); the real-QLoRA-through-OpenDSS subset is
in the supplementary material.

Run: python3 -m code.evaluation.final_safeagent_v3.run_opendss_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..physical_curves import sweep
from ..opendss_sweep.run_opendss_sweep import (IEEE13_DIR, IEEE34_DIR,
                                               build_configs)
from ..final_safeagent.run_p1a_opendss import SUBSET
from ...simulation.feeder import OpenDSSFeeder
from ...simulation.harness import run_scenario

OUT = Path("code/results/ijcip_final/opendss_check")


def _mk(arm, cfg):
    if arm == "D0":
        from ...baselines.deterministic_energy_policy.adapter import (
            DeterministicEnergyPolicy)
        det = DeterministicEnergyPolicy(runtime_safe=True)
    else:  # OPROJ
        from ..final_safeagent.run_p1b_projection import OracleProjectionDetector
        det = OracleProjectionDetector()
        det.configure(cfg.get("ground_truth") or {})
    det.name = arm
    return det


def main():
    import opendssdirect  # noqa: F401 — fail loudly if absent
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {c["id"]: c for c in build_configs()}
    rows = []
    for arm in ("D0", "OPROJ"):
        for sid in SUBSET:
            c = configs[sid]
            feeder_dir = IEEE13_DIR if c["family"] == "ieee13" else IEEE34_DIR
            cfg = {"name": sid, "description": c["why"],
                   "feeder": str((feeder_dir / "feeder.dss").resolve()),
                   "dt_s": 1.0, "duration_s": 600, "monitored_buses": c["buses"],
                   "ders": c["ders"], "attacks": c["attacks"],
                   "ground_truth": {"attack_class": c["gt_class"],
                                    "affected_asset": c["gt_asset"],
                                    "start_s": (c["attacks"][0]["start_s"] if c["attacks"] else 0),
                                    "end_s": (c["attacks"][0]["end_s"] if c["attacks"] else 0)},
                   "opendss_init_commands": c["init_commands"]}
            scen_dir = OUT / "scenarios" / sid
            scen_dir.mkdir(parents=True, exist_ok=True)
            p = scen_dir / "config.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            root = OUT / "runs" / arm
            rd = root / sid / arm / "seed0"
            if not (rd / "manifest.json").exists():
                feeder = OpenDSSFeeder(dss_path=cfg["feeder"],
                                       monitored_buses=c["buses"], ders=c["ders"],
                                       init_commands=c["init_commands"])
                det = _mk(arm, cfg)
                run_scenario(str(p), det, seed=0, out_root=str(root), feeder=feeder,
                             extra_manifest={"tag": "ijcip_final", "arm": arm,
                                             "runtime_safe": True,
                                             "backend": "opendss_real",
                                             "n_nonconverged": feeder.n_nonconverged})
                mp = rd / "manifest.json"
                mm = json.loads(mp.read_text())
                mm["n_nonconverged"] = feeder.n_nonconverged
                mp.write_text(json.dumps(mm, indent=2))
            mm = json.loads((rd / "manifest.json").read_text())
            ph = sweep(rd, np.linspace(0, 1, 11))
            a = ph[ph.threshold.between(0.49, 0.51)].iloc[0]
            ts = pd.read_csv(rd / "timeseries.csv")
            vcols = [x for x in ts.columns if x.startswith("v_pu_")]
            rows.append({"arm": arm, "config": sid, "feeder": c["family"],
                         "attack_type": c["gt_class"],
                         "ens_kwh": float(a["ens_kwh"]),
                         "curt_kwh": float(a["curt_kwh"]),
                         "vmin_pu": float(ts[vcols].min().min()),
                         "vviol_frac": float(a["voltage_frac"]),
                         "n_nonconverged": int(mm.get("n_nonconverged", 0))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "opendss_check_raw.csv", index=False)
    summ = (df[df.attack_type != "none"].groupby("arm")
            .agg(ens=("ens_kwh", "mean"), curt=("curt_kwh", "mean"),
                 vmin=("vmin_pu", "min"), nonconv=("n_nonconverged", "sum"),
                 n=("config", "count")).reset_index())
    summ.to_csv(OUT / "opendss_check_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print("total non-converged solves:", int(df.n_nonconverged.sum()))


if __name__ == "__main__":
    main()
