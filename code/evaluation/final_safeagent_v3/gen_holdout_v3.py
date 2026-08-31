"""V3 Phase 2 — generate and HASH-FREEZE the independent end-to-end holdout.

This holdout is for end-to-end SYSTEM evaluation only. It is causally distinct
from the 25-config development library (new start times, durations, magnitudes,
attacked buses/assets, load and penetration points, benign near-boundary
conditions) and from the component-level estimator/gate holdouts. It is frozen
(per-config SHA-256 + manifest hash + timestamped freeze record) BEFORE any
model or policy is executed on it; the frozen analysis plan is written
alongside. IEEE-123 is unavailable in the repository (documented, not
silently substituted): the holdout uses IEEE-13 and IEEE-34.

Coverage: 4 attack families (fdi, command_spoof, replay, dos) + a fleet
(multi-asset) attack, 3 magnitudes, 3 durations, 3 load levels, 3 penetration
levels, multiple attacked buses across two feeders, plus 12+ benign /
near-boundary conditions (cloud transients, measurement noise, historian
echo). 3 genuine stochastic episodes per configuration.

Run: python3 -m code.evaluation.final_safeagent_v3.gen_holdout_v3
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

CONF = Path("code/configs/ijcip_final_v3")
SCEN = Path("code/simulation/scenarios/ijcip_final_v3")
IEEE13_BUSES = ["SOURCEBUS", "634", "671", "675"]
IEEE34_BUSES = ["SOURCEBUS", "808", "824", "834", "840", "848"]
N_EPISODES = 3
EP_SEEDS = [1301, 1302, 1303]   # frozen episode seeds


def ders13(scale=1.0):
    return [{"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0 * scale},
            {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0 * scale}]


def ders34():
    return [{"id": "INV_840", "type": "pv", "bus": "840", "max_kw": 240.0},
            {"id": "BESS_848", "type": "bess", "bus": "848", "max_kw": 180.0},
            {"id": "INV_824", "type": "pv", "bus": "824", "max_kw": 260.0}]


def _cfg(name, family, attacks, gt_class, gt_asset, load_kw, ders, why,
         stub=None):
    a0 = attacks[0] if attacks else {"start_s": 0, "end_s": 0}
    return dict(
        name=name, description=why,
        feeder=("../../ieee13_fdi_inverter/feeder.dss" if family == "ieee13"
                else "../../ieee34_command_spoof_derms/feeder.dss"),
        dt_s=1.0, duration_s=600,
        monitored_buses=(IEEE13_BUSES if family == "ieee13" else IEEE34_BUSES),
        ders=ders, base_load_kw=float(load_kw), attacks=attacks,
        ground_truth=dict(attack_class=gt_class, affected_asset=gt_asset,
                          start_s=a0["start_s"], end_s=a0["end_s"]),
        stub_params=stub or {},
        revision_meta=dict(tag="ijcip_final_v3", split="end_to_end_holdout",
                           feeder_family=family))


def build():
    C = []
    # ---- attacks: spread over new magnitudes/durations/starts/buses -------
    # FDI (INV_634 / BESS / IEEE34), 3 magnitudes incl. high (observable)
    for mag, bias, dur, st in (("low", 18.0, 90, 150), ("med", 55.0, 240, 130),
                               ("high", 130.0, 300, 120)):
        C.append(_cfg(f"h_fdi_{mag}_13", "ieee13",
                      [dict(type="fdi", target="INV_634", start_s=st,
                            end_s=st + dur, bias_kw=bias)],
                      "fdi", "INV_634", 1000.0, ders13(),
                      f"FDI {mag} bias {bias} kW, dur {dur}s"))
    C.append(_cfg("h_fdi_high_bess_13", "ieee13",
                  [dict(type="fdi", target="BESS_675", start_s=140, end_s=440,
                        bias_kw=110.0)], "fdi", "BESS_675", 1100.0, ders13(),
                  "FDI high on storage unit, elevated load"))
    C.append(_cfg("h_fdi_med_34", "ieee34",
                  [dict(type="fdi", target="INV_824", start_s=160, end_s=400,
                        bias_kw=90.0)], "fdi", "INV_824", 1000.0, ders34(),
                  "FDI on IEEE-34 mid-feeder PV"))
    # command spoof, 3 magnitudes, durations, buses
    for mag, kw, dur, st in (("low", 110.0, 90, 200), ("med", 45.0, 240, 150),
                             ("high", 5.0, 300, 130)):
        C.append(_cfg(f"h_spoof_{mag}_13", "ieee13",
                      [dict(type="command_spoof", target="INV_634", start_s=st,
                            end_s=st + dur, forged_setpoint_kw=kw, period_s=5.0)],
                      "command_spoof", "INV_634", 1000.0, ders13(),
                      f"spoof {mag} to {kw} kW"))
    C.append(_cfg("h_spoof_bess_13", "ieee13",
                  [dict(type="command_spoof", target="BESS_675", start_s=150,
                        end_s=450, forged_setpoint_kw=15.0, period_s=5.0)],
                  "command_spoof", "BESS_675", 1250.0, ders13(0.85),
                  "spoof storage, peak load, low penetration"))
    C.append(_cfg("h_spoof_34_distal", "ieee34",
                  [dict(type="command_spoof", target="INV_840", start_s=170,
                        end_s=410, forged_setpoint_kw=10.0, period_s=5.0)],
                  "command_spoof", "INV_840", 1000.0, ders34(),
                  "spoof most-distal PV on the long radial"))
    # replay, durations/buses
    for dur, st, cap in ((90, 220, 15.0), (240, 140, 45.0), (300, 130, 60.0)):
        C.append(_cfg(f"h_replay_{dur}_13", "ieee13",
                      [dict(type="replay", target="INV_634", start_s=st,
                            end_s=st + dur, capture_window_s=cap)],
                      "replay", "INV_634", 1000.0, ders13(),
                      f"replay dur {dur}s"))
    C.append(_cfg("h_replay_34", "ieee34",
                  [dict(type="replay", target="BESS_848", start_s=160,
                        end_s=400, capture_window_s=30.0)],
                  "replay", "BESS_848", 1000.0, ders34(), "replay IEEE-34 BESS"))
    # dos, durations/buses/loads
    for dur, st, load in ((60, 250, 1000.0), (240, 150, 1400.0), (300, 130, 1000.0)):
        C.append(_cfg(f"h_dos_{dur}_13", "ieee13",
                      [dict(type="dos", target="INV_634", start_s=st,
                            end_s=st + dur)], "dos", "INV_634", load, ders13(),
                      f"DoS dur {dur}s, load {load}"))
    C.append(_cfg("h_dos_bess_34", "ieee34",
                  [dict(type="dos", target="BESS_848", start_s=150, end_s=450)],
                  "dos", "BESS_848", 1000.0, ders34(), "DoS IEEE-34 storage"))
    # fleet / multi-asset
    C.append(_cfg("h_fleet_spoof_34", "ieee34",
                  [dict(type="command_spoof", target="INV_840", start_s=150,
                        end_s=390, forged_setpoint_kw=12.0, period_s=5.0),
                   dict(type="command_spoof", target="INV_824", start_s=150,
                        end_s=390, forged_setpoint_kw=12.0, period_s=5.0)],
                  "command_spoof", "INV_840", 1000.0, ders34(),
                  "coordinated two-unit fleet spoof"))
    C.append(_cfg("h_fleet_fdi_34", "ieee34",
                  [dict(type="fdi", target="INV_840", start_s=160, end_s=400,
                        bias_kw=150.0),
                   dict(type="fdi", target="INV_824", start_s=160, end_s=400,
                        bias_kw=150.0)], "fdi", "INV_840", 1000.0, ders34(),
                  "coordinated two-unit fleet FDI (high magnitude)"))
    # penetration / load sweep on spoof
    for pen, load, tag in ((0.5, 1000.0, "lowpen"), (1.5, 1000.0, "highpen"),
                           (1.0, 700.0, "lightload")):
        C.append(_cfg(f"h_spoof_{tag}_13", "ieee13",
                      [dict(type="command_spoof", target="INV_634", start_s=160,
                            end_s=400, forged_setpoint_kw=30.0, period_s=5.0)],
                      "command_spoof", "INV_634", load, ders13(pen),
                      f"spoof {tag}"))
    # penetration / load sweep on FDI-high and DoS (fills feeder x family x op-point)
    for pen, load, tag in ((0.5, 1000.0, "lowpen"), (1.5, 1000.0, "highpen"),
                           (1.0, 1400.0, "peakload")):
        C.append(_cfg(f"h_fdi_high_{tag}_13", "ieee13",
                      [dict(type="fdi", target="INV_634", start_s=150,
                            end_s=390, bias_kw=130.0)], "fdi", "INV_634", load,
                      ders13(pen), f"FDI high {tag}"))
    for pen, load, tag in ((0.5, 1000.0, "lowpen"), (1.5, 1000.0, "highpen")):
        C.append(_cfg(f"h_dos_{tag}_13", "ieee13",
                      [dict(type="dos", target="INV_634", start_s=160,
                            end_s=400)], "dos", "INV_634", load, ders13(pen),
                      f"DoS {tag}"))
    # replay at additional buses / penetration
    C.append(_cfg("h_replay_lowpen_13", "ieee13",
                  [dict(type="replay", target="INV_634", start_s=160, end_s=400,
                        capture_window_s=30.0)], "replay", "INV_634", 1000.0,
                  ders13(0.5), "replay low penetration"))
    C.append(_cfg("h_replay_bess_13", "ieee13",
                  [dict(type="replay", target="BESS_675", start_s=170, end_s=410,
                        capture_window_s=25.0)], "replay", "BESS_675", 1000.0,
                  ders13(), "replay on storage unit"))
    # short-duration near-boundary attacks (detection-difficulty axis)
    C.append(_cfg("h_spoof_short_13", "ieee13",
                  [dict(type="command_spoof", target="INV_634", start_s=260,
                        end_s=320, forged_setpoint_kw=40.0, period_s=5.0)],
                  "command_spoof", "INV_634", 1000.0, ders13(), "60 s spoof"))
    C.append(_cfg("h_dos_short_13", "ieee13",
                  [dict(type="dos", target="INV_634", start_s=270, end_s=320)],
                  "dos", "INV_634", 1000.0, ders13(), "50 s DoS"))
    C.append(_cfg("h_fleet_dos_34", "ieee34",
                  [dict(type="dos", target="INV_840", start_s=160, end_s=400),
                   dict(type="dos", target="BESS_848", start_s=160, end_s=400)],
                  "dos", "INV_840", 1000.0, ders34(), "coordinated fleet DoS"))

    # ---- benign / near-boundary (>=12) ------------------------------------
    benign_specs = [
        ("h_benign_norm_0", 1000.0, 1.0, 0.08, 0.004, 0.8),
        ("h_benign_norm_1", 800.0, 0.8, 0.06, 0.003, 0.6),
        ("h_benign_norm_2", 1300.0, 1.3, 0.10, 0.006, 1.2),
        ("h_benign_norm_3", 1100.0, 0.9, 0.09, 0.005, 1.0),
        ("h_benign_clouds_0", 1000.0, 1.0, 0.12, 0.025, 2.5),
        ("h_benign_clouds_1", 1350.0, 1.2, 0.15, 0.030, 3.0),
        ("h_benign_noise_0", 1000.0, 1.0, 0.10, 0.008, 5.0),
        ("h_benign_noise_1", 900.0, 0.9, 0.12, 0.010, 7.0),
        ("h_benign_mixed", 1150.0, 1.1, 0.14, 0.020, 4.0),
        ("h_benign_norm_4", 950.0, 1.0, 0.07, 0.004, 0.9),
        ("h_benign_norm_5", 1200.0, 1.2, 0.10, 0.005, 1.1),
        ("h_benign_lightload", 700.0, 0.7, 0.05, 0.002, 0.5),
        ("h_benign_clouds_2", 1000.0, 1.0, 0.13, 0.028, 2.8),
    ]
    for i, (name, load, pen, ls, cr, nk) in enumerate(benign_specs):
        c = _cfg(name, "ieee13", [], "none", None, load, ders13(pen),
                 "benign near-boundary",
                 stub=dict(episode_seed=1400 + i, load_sigma=ls,
                           pv_cloud_rate=cr, meas_noise_kw=nk))
        C.append(c)
    # historian echo (benign protocol ambiguity)
    for i, (per, s0, s1) in enumerate([(3.0, 120, 480), (2.0, 150, 450),
                                       (5.0, 100, 500)]):
        c = _cfg(f"h_benign_echo_{i}", "ieee13",
                 [dict(type="benign_echo", target="INV_634", start_s=s0,
                       end_s=s1, period_s=per)], "none", None, 1000.0, ders13(),
                 "benign historian echo",
                 stub=dict(episode_seed=1420 + i))
        C.append(c)
    assert len({c["name"] for c in C}) == len(C)
    return C


def main():
    CONF.mkdir(parents=True, exist_ok=True)
    SCEN.mkdir(parents=True, exist_ok=True)
    (SCEN / "__init__.py").touch()
    configs = build()
    rows = []
    for c in configs:
        d = SCEN / c["name"]
        d.mkdir(exist_ok=True)
        p = d / "config.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False))
        gt = c["ground_truth"]
        rows.append(dict(
            scenario_id=c["name"],
            kind=("benign" if gt["attack_class"] == "none" else "attack"),
            attack_type=gt["attack_class"], affected_asset=gt["affected_asset"],
            start_s=gt["start_s"], end_s=gt["end_s"],
            duration_s=gt["end_s"] - gt["start_s"],
            base_load_kw=c["base_load_kw"],
            feeder_family=c["revision_meta"]["feeder_family"],
            n_episodes=N_EPISODES, episode_seeds=";".join(map(str, EP_SEEDS)),
            config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            config_path=str(p.resolve())))
    df = pd.DataFrame(rows)
    man = CONF / "holdout_v3_manifest.csv"
    df.to_csv(man, index=False)
    n_atk = int((df.kind == "attack").sum())
    n_ben = int((df.kind == "benign").sum())
    freeze = {
        "frozen_at": "2026-08-11",
        "tag": "ijcip_final_v3",
        "purpose": "end-to-end system evaluation ONLY; causally distinct from "
                   "the 25-config development library and the component holdouts",
        "n_configurations": len(rows), "n_attack": n_atk, "n_benign": n_ben,
        "n_episodes_per_config": N_EPISODES, "episode_seeds": EP_SEEDS,
        "feeders": ["ieee13", "ieee34"],
        "ieee123": "UNAVAILABLE in repository; not substituted (documented "
                   "reduced feeder scope)",
        "attack_families": ["fdi", "command_spoof", "replay", "dos",
                            "fleet(multi-asset)"],
        "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
        "rule": "frozen BEFORE any model/policy executed on it; runtime path "
                "is runtime_safe (no tampered oracle); ground truth used only "
                "by the offline evaluator after action selection",
    }
    (CONF / "holdout_v3_freeze.json").write_text(json.dumps(freeze, indent=2))
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
