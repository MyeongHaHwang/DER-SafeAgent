"""P0-A3: generate and FREEZE the estimator dev/holdout manifests.

- DEVELOPMENT = the existing frozen 25-configuration library (these informed
  the estimator diagnosis and the EH design, so they cannot test the repair).
- HELD-OUT = 12 NEW configurations, causally different from every dev
  configuration (new magnitudes, durations {45,240,420} s, start times,
  targets incl. BESS and IEEE-34 mid-feeder PV, load levels 850/1100 kW, a
  0.85 penetration level, and a two-asset fleet attack).

Also freezes the EH/EMH duration-prior calibration from DEV ground truths
only. Everything is written BEFORE any estimator is evaluated on the holdout;
SHA-256 hashes of each config file are recorded in the manifest.

Run: python3 -m code.evaluation.final_safeagent.gen_holdout_configs
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from ...Multi_AI_Agent.horizon_estimator import HorizonCalibration

TAG = "ijcip_final_safeagent_20260810"
CONF = Path("code/configs") / TAG
SCEN = Path("code/simulation/scenarios/ijcip_final_safeagent")
DEV_MANIFEST = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")

IEEE13_BUSES = ["SOURCEBUS", "634", "671", "675"]
IEEE34_BUSES = ["SOURCEBUS", "808", "824", "834", "840", "848"]


def ieee13_ders(scale: float = 1.0) -> list[dict]:
    return [{"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0 * scale},
            {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0 * scale}]


def ieee34_ders() -> list[dict]:
    return [{"id": "INV_840", "type": "pv", "bus": "840", "max_kw": 240.0},
            {"id": "BESS_848", "type": "bess", "bus": "848", "max_kw": 180.0},
            {"id": "INV_824", "type": "pv", "bus": "824", "max_kw": 260.0}]


def _cfg(id_, family, attacks, gt_class, gt_asset, load_kw, ders, why):
    a0 = attacks[0]
    return dict(
        name=id_, description=why,
        feeder=("../../ieee13_fdi_inverter/feeder.dss" if family == "ieee13"
                else "../../ieee34_command_spoof_derms/feeder.dss"),
        dt_s=1.0, duration_s=600,
        monitored_buses=(IEEE13_BUSES if family == "ieee13" else IEEE34_BUSES),
        ders=ders, base_load_kw=float(load_kw), attacks=attacks,
        ground_truth=dict(attack_class=gt_class, affected_asset=gt_asset,
                          start_s=a0["start_s"], end_s=a0["end_s"]),
        revision_meta=dict(revision_tag=TAG, split="impact_estimator_holdout",
                           feeder_family=family))


def build_holdout() -> list[dict]:
    C = []
    # 1 sub-threshold spoof, short, early start — near the no_op/freeze boundary
    C.append(_cfg("ho13_spoof_sub_short", "ieee13",
                  [dict(type="command_spoof", target="INV_634", start_s=150,
                        end_s=195, forged_setpoint_kw=120.0, period_s=5.0)],
                  "command_spoof", "INV_634", 1000.0, ieee13_ders(),
                  "Sub-threshold 45 s spoof (deficit ~0.07): near-boundary case."))
    # 2 extreme spoof, long, elevated load
    C.append(_cfg("ho13_spoof_zero_long", "ieee13",
                  [dict(type="command_spoof", target="INV_634", start_s=120,
                        end_s=540, forged_setpoint_kw=0.0, period_s=5.0)],
                  "command_spoof", "INV_634", 1100.0, ieee13_ders(),
                  "Forced-zero 420 s spoof at 1.1x load: deepest supra-threshold case."))
    # 3 spoof on the BESS (dev spoofs the BESS only once, different magnitude/dur)
    C.append(_cfg("ho13_spoof_bess_240", "ieee13",
                  [dict(type="command_spoof", target="BESS_675", start_s=180,
                        end_s=420, forged_setpoint_kw=20.0, period_s=5.0)],
                  "command_spoof", "BESS_675", 1000.0, ieee13_ders(),
                  "240 s storage-unit spoof, new magnitude/duration."))
    # 4 extreme FDI, long
    C.append(_cfg("ho13_fdi_extreme_long", "ieee13",
                  [dict(type="fdi", target="INV_634", start_s=120, end_s=540,
                        bias_kw=150.0)],
                  "fdi", "INV_634", 1000.0, ieee13_ders(),
                  "420 s FDI with bias outside the dev range: over-mitigation trap."))
    # 5 low FDI, short
    C.append(_cfg("ho13_fdi_low_short", "ieee13",
                  [dict(type="fdi", target="INV_634", start_s=240, end_s=285,
                        bias_kw=15.0)],
                  "fdi", "INV_634", 1000.0, ieee13_ders(),
                  "45 s near-noise FDI, late start."))
    # 6 short DoS, late start
    C.append(_cfg("ho13_dos_short_late", "ieee13",
                  [dict(type="dos", target="INV_634", start_s=240, end_s=285)],
                  "dos", "INV_634", 1000.0, ieee13_ders(),
                  "45 s outage: stale-family case where over-committing is cheap."))
    # 7 long DoS on the BESS
    C.append(_cfg("ho13_dos_bess_long", "ieee13",
                  [dict(type="dos", target="BESS_675", start_s=120, end_s=540)],
                  "dos", "BESS_675", 1000.0, ieee13_ders(),
                  "420 s storage outage: stale family, maximal horizon."))
    # 8 long replay
    C.append(_cfg("ho13_replay_long", "ieee13",
                  [dict(type="replay", target="INV_634", start_s=120, end_s=540,
                        capture_window_s=45.0)],
                  "replay", "INV_634", 1000.0, ieee13_ders(),
                  "420 s replay: stale family where the physical state is healthy."))
    # 9 IEEE-34 mid-feeder PV spoof (asset not attacked in dev)
    C.append(_cfg("ho34_spoof_inv824_240", "ieee34",
                  [dict(type="command_spoof", target="INV_824", start_s=150,
                        end_s=390, forged_setpoint_kw=20.0, period_s=5.0)],
                  "command_spoof", "INV_824", 1000.0, ieee34_ders(),
                  "Mid-feeder PV spoof on the long radial, 240 s."))
    # 10 IEEE-34 FDI on the storage unit (class x feeder cell absent from dev)
    C.append(_cfg("ho34_fdi_bess848_240", "ieee34",
                  [dict(type="fdi", target="BESS_848", start_s=150, end_s=390,
                        bias_kw=80.0)],
                  "fdi", "BESS_848", 1000.0, ieee34_ders(),
                  "FDI against IEEE-34 storage: new class x feeder x asset cell."))
    # 11 two-asset fleet spoof, 240 s
    C.append(_cfg("ho34_fleet_spoof_240", "ieee34",
                  [dict(type="command_spoof", target="INV_840", start_s=150,
                        end_s=390, forged_setpoint_kw=15.0, period_s=5.0),
                   dict(type="command_spoof", target="INV_824", start_s=150,
                        end_s=390, forged_setpoint_kw=15.0, period_s=5.0)],
                  "command_spoof", "INV_840", 1000.0, ieee34_ders(),
                  "Coordinated two-unit spoof (DERMS-compromise proxy), 240 s."))
    # 12 new load and penetration point
    C.append(_cfg("ho13_spoof_load085_pen085", "ieee13",
                  [dict(type="command_spoof", target="INV_634", start_s=180,
                        end_s=420, forged_setpoint_kw=25.0, period_s=5.0)],
                  "command_spoof", "INV_634", 850.0, ieee13_ders(0.85),
                  "240 s spoof at 0.85x load and 0.85x penetration: new operating point."))
    assert len({c["name"] for c in C}) == 12
    return C


def main() -> None:
    CONF.mkdir(parents=True, exist_ok=True)
    SCEN.mkdir(parents=True, exist_ok=True)
    (SCEN / "__init__.py").touch()

    # ---- holdout configs + frozen manifest -------------------------------
    rows = []
    for c in build_holdout():
        d = SCEN / c["name"]
        d.mkdir(exist_ok=True)
        p = d / "config.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False))
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        gt = c["ground_truth"]
        rows.append(dict(scenario_id=c["name"], split="holdout",
                         attack_type=gt["attack_class"],
                         affected_asset=gt["affected_asset"],
                         start_s=gt["start_s"], end_s=gt["end_s"],
                         duration_s=gt["end_s"] - gt["start_s"],
                         base_load_kw=c["base_load_kw"],
                         feeder_family=c["revision_meta"]["feeder_family"],
                         config_sha256=h, config_path=str(p.resolve())))
    pd.DataFrame(rows).to_csv(CONF / "impact_estimator_holdout.csv", index=False)

    # ---- dev manifest pointer (existing frozen library) ------------------
    dev = pd.read_csv(DEV_MANIFEST)
    dev.insert(1, "split", "dev")
    dev.to_csv(CONF / "impact_estimator_dev.csv", index=False)

    # ---- EH/EMH duration prior: DEV ground truths only, frozen now -------
    fam_map = {"command_spoof": "spoof_like", "fdi": "fdi_like",
               "replay": "stale", "dos": "stale"}
    per_class: dict[str, list[float]] = {}
    for _, r in dev.iterrows():
        cfg = yaml.safe_load(Path(r.config_path).read_text())
        gt = cfg.get("ground_truth") or {}
        cls = gt.get("attack_class", "none")
        if cls in fam_map:
            per_class.setdefault(fam_map[cls], []).append(
                float(gt["end_s"]) - float(gt["start_s"]))
    calib = HorizonCalibration.from_durations(per_class)
    calib.save(CONF / "eh_duration_prior.json")

    freeze_note = {
        "frozen_at": "2026-08-10",
        "rule": ("holdout manifest, configs and EH/EMH duration prior frozen "
                 "BEFORE any of E60/EH/EMH was evaluated on the holdout; the "
                 "prior uses DEV ground-truth durations only"),
        "n_dev": int(len(dev)), "n_holdout": len(rows),
        "manifest_sha256": hashlib.sha256(
            (CONF / "impact_estimator_holdout.csv").read_bytes()).hexdigest(),
        "prior_sha256": hashlib.sha256(
            (CONF / "eh_duration_prior.json").read_bytes()).hexdigest(),
    }
    (CONF / "impact_estimator_freeze.json").write_text(
        json.dumps(freeze_note, indent=2))
    print(json.dumps(freeze_note, indent=2))


if __name__ == "__main__":
    main()
