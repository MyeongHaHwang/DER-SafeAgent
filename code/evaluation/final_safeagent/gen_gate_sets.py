"""P0-B: generate the Evidence-Gate DEVELOPMENT and FROZEN TEST sets.

DEVELOPMENT (threshold calibration only):
  - 4 benign-normal + 4 benign-suspicious stochastic configurations
    (calibration-specific exogenous parameters; distinct from every test
    configuration), plus the 24 existing dev attack configurations.

FROZEN TEST (evaluated once, after thresholds are frozen):
  - 8 benign-normal (varied load level, penetration, cloud rate, noise)
  - 8 benign-but-suspicious near the decision boundary, including THREE
    configurations with a benign protocol echo (duplicated telemetry events
    with no tampering and no physical effect) so hard-evidence ambiguity
    exists inside benign data
  - 16 NEW attack configurations, 4 per class, including low-magnitude
    near-boundary cases; parameters differ from every dev configuration and
    from the P0-A estimator holdout.

All configs are written under code/simulation/scenarios/ijcip_final_safeagent/
gate_* and hashed into code/configs/ijcip_final_safeagent_20260810/
evidence_gate_{dev,test}.csv BEFORE calibration/evaluation respectively.

Run: python3 -m code.evaluation.final_safeagent.gen_gate_sets
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

TAG = "ijcip_final_safeagent_20260810"
CONF = Path("code/configs") / TAG
SCEN = Path("code/simulation/scenarios/ijcip_final_safeagent")
DEV_ATTACKS = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")

IEEE13_BUSES = ["SOURCEBUS", "634", "671", "675"]


def ders13(scale: float = 1.0) -> list[dict]:
    return [{"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0 * scale},
            {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0 * scale}]


def _benign(id_, load_kw, pen, seed, load_sigma, cloud_rate, noise_kw, why,
            attacks=None):
    return dict(
        name=id_, description=why, feeder="../../ieee13_fdi_inverter/feeder.dss",
        dt_s=1.0, duration_s=600, monitored_buses=IEEE13_BUSES,
        ders=ders13(pen), base_load_kw=float(load_kw), attacks=attacks or [],
        ground_truth=dict(attack_class="none", affected_asset=None,
                          start_s=0, end_s=0),
        stub_params=dict(episode_seed=seed, load_sigma=load_sigma,
                         pv_cloud_rate=cloud_rate, meas_noise_kw=noise_kw),
        revision_meta=dict(revision_tag=TAG, split="evidence_gate",
                           condition_kind=("benign_suspicious"
                                           if (cloud_rate > 0.01 or noise_kw > 2.0
                                               or attacks) else "benign_normal")))


_ATTACK_SEED = [400]


def _attack(id_, attacks, gt_class, gt_asset, load_kw=1000.0, pen=1.0, why=""):
    a0 = attacks[0]
    _ATTACK_SEED[0] += 1
    return dict(
        name=id_, description=why, feeder="../../ieee13_fdi_inverter/feeder.dss",
        dt_s=1.0, duration_s=600, monitored_buses=IEEE13_BUSES,
        ders=ders13(pen), base_load_kw=float(load_kw), attacks=attacks,
        ground_truth=dict(attack_class=gt_class, affected_asset=gt_asset,
                          start_s=a0["start_s"], end_s=a0["end_s"]),
        # Genuine exogenous stochasticity (published parameters). Noise-free
        # deterministic telemetry makes any staleness/duplicate signature
        # degenerate (every payload repeats exactly), which the first
        # calibration pass exposed; all gate-set configs are therefore
        # stochastic episodes.
        stub_params=dict(episode_seed=_ATTACK_SEED[0], load_sigma=0.08,
                         pv_cloud_rate=0.004, meas_noise_kw=0.8),
        revision_meta=dict(revision_tag=TAG, split="evidence_gate",
                           condition_kind="attack"))


def build_dev_benign() -> list[dict]:
    return [
        _benign("gd_benign_norm_a", 1000, 1.0, 101, 0.08, 0.004, 0.8,
                "dev benign-normal, published exogenous parameters"),
        _benign("gd_benign_norm_b", 800, 0.8, 102, 0.06, 0.003, 0.6,
                "dev benign-normal, light load / low penetration"),
        _benign("gd_benign_norm_c", 1250, 1.2, 103, 0.10, 0.005, 1.0,
                "dev benign-normal, heavy load / high penetration"),
        _benign("gd_benign_norm_d", 1000, 1.0, 104, 0.08, 0.004, 0.8,
                "dev benign-normal, second draw of published parameters"),
        _benign("gd_benign_susp_a", 1000, 1.0, 111, 0.12, 0.02, 3.0,
                "dev benign-suspicious: frequent clouds + heavy noise"),
        _benign("gd_benign_susp_b", 1000, 1.0, 112, 0.15, 0.03, 5.0,
                "dev benign-suspicious: extreme noise bursts"),
        _benign("gd_benign_susp_c", 1400, 1.0, 113, 0.12, 0.025, 2.5,
                "dev benign-suspicious: peak load + frequent deep clouds"),
        _benign("gd_benign_susp_echo", 1000, 1.0, 114, 0.08, 0.004, 0.8,
                "dev benign-suspicious: historian echo (duplicated events)",
                attacks=[dict(type="benign_echo", target="INV_634",
                              start_s=120, end_s=480, period_s=3.0)]),
    ]


def build_test() -> list[dict]:
    C: list[dict] = []
    # --- 8 benign-normal (causally varied exogenous conditions) ----------
    params = [(1000, 1.0, 0.08, 0.004, 0.8), (700, 0.7, 0.05, 0.002, 0.5),
              (1300, 1.3, 0.10, 0.006, 1.2), (900, 1.0, 0.07, 0.004, 0.8),
              (1100, 0.9, 0.09, 0.005, 1.0), (1000, 1.1, 0.08, 0.003, 0.7),
              (850, 1.0, 0.06, 0.004, 0.9), (1200, 1.2, 0.10, 0.005, 1.1)]
    for i, (load, pen, ls, cr, nk) in enumerate(params):
        C.append(_benign(f"gt_benign_norm_{i}", load, pen, 200 + i, ls, cr, nk,
                         f"test benign-normal variant {i}"))
    # --- 8 benign-suspicious ---------------------------------------------
    C.append(_benign("gt_susp_clouds_a", 1000, 1.0, 301, 0.12, 0.025, 2.5,
                     "test: frequent deep clouds"))
    C.append(_benign("gt_susp_clouds_b", 1350, 1.2, 302, 0.15, 0.03, 3.0,
                     "test: peak load + storms"))
    C.append(_benign("gt_susp_noise_a", 1000, 1.0, 303, 0.10, 0.008, 5.0,
                     "test: heavy measurement noise"))
    C.append(_benign("gt_susp_noise_b", 900, 0.9, 304, 0.12, 0.01, 7.0,
                     "test: extreme measurement noise"))
    C.append(_benign("gt_susp_mixed", 1150, 1.1, 305, 0.14, 0.02, 4.0,
                     "test: clouds + noise combined"))
    for i, (per, s0, s1) in enumerate([(3.0, 100, 500), (2.0, 150, 450),
                                       (5.0, 60, 540)]):
        C.append(_benign(f"gt_susp_echo_{i}", 1000, 1.0, 310 + i, 0.08, 0.004, 0.8,
                         "test: benign historian echo (hard-evidence ambiguity)",
                         attacks=[dict(type="benign_echo", target="INV_634",
                                       start_s=s0, end_s=s1, period_s=per)]))
    # --- 16 attacks: 4 per class, new parameters -------------------------
    C += [
        _attack("gt_fdi_near_noise", [dict(type="fdi", target="INV_634",
                start_s=200, end_s=380, bias_kw=10.0)], "fdi", "INV_634",
                why="near-noise FDI bias 10 kW: boundary case"),
        _attack("gt_fdi_mid", [dict(type="fdi", target="INV_634",
                start_s=160, end_s=340, bias_kw=45.0)], "fdi", "INV_634"),
        _attack("gt_fdi_high_bess", [dict(type="fdi", target="BESS_675",
                start_s=140, end_s=380, bias_kw=100.0)], "fdi", "BESS_675"),
        _attack("gt_fdi_late_short", [dict(type="fdi", target="INV_634",
                start_s=420, end_s=510, bias_kw=70.0)], "fdi", "INV_634"),
        _attack("gt_spoof_mild", [dict(type="command_spoof", target="INV_634",
                start_s=200, end_s=380, forged_setpoint_kw=110.0, period_s=5.0)],
                "command_spoof", "INV_634",
                why="mild spoof (deficit 0.11): boundary case"),
        _attack("gt_spoof_mid", [dict(type="command_spoof", target="INV_634",
                start_s=160, end_s=400, forged_setpoint_kw=50.0, period_s=5.0)],
                "command_spoof", "INV_634"),
        _attack("gt_spoof_deep_bess", [dict(type="command_spoof", target="BESS_675",
                start_s=140, end_s=440, forged_setpoint_kw=10.0, period_s=5.0)],
                "command_spoof", "BESS_675"),
        _attack("gt_spoof_slow_period", [dict(type="command_spoof", target="INV_634",
                start_s=180, end_s=420, forged_setpoint_kw=35.0, period_s=17.0)],
                "command_spoof", "INV_634",
                why="slow-period spoof: sparse hard evidence"),
        _attack("gt_replay_short", [dict(type="replay", target="INV_634",
                start_s=220, end_s=310, capture_window_s=15.0)], "replay", "INV_634"),
        _attack("gt_replay_long", [dict(type="replay", target="INV_634",
                start_s=130, end_s=470, capture_window_s=60.0)], "replay", "INV_634"),
        _attack("gt_replay_bess", [dict(type="replay", target="BESS_675",
                start_s=180, end_s=360, capture_window_s=25.0)], "replay", "BESS_675"),
        _attack("gt_replay_low_pen", [dict(type="replay", target="INV_634",
                start_s=200, end_s=380, capture_window_s=20.0)], "replay",
                "INV_634", pen=0.6),
        _attack("gt_dos_short", [dict(type="dos", target="INV_634",
                start_s=250, end_s=310)], "dos", "INV_634"),
        _attack("gt_dos_long", [dict(type="dos", target="INV_634",
                start_s=130, end_s=470)], "dos", "INV_634"),
        _attack("gt_dos_bess", [dict(type="dos", target="BESS_675",
                start_s=180, end_s=390)], "dos", "BESS_675"),
        _attack("gt_dos_peakload", [dict(type="dos", target="INV_634",
                start_s=200, end_s=400)], "dos", "INV_634", load_kw=1400.0),
    ]
    assert len({c["name"] for c in C}) == 32
    return C


def _write(configs: list[dict], manifest_path: Path) -> None:
    rows = []
    for c in configs:
        d = SCEN / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / "config.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False))
        gt = c["ground_truth"]
        rows.append(dict(
            scenario_id=c["name"],
            condition_kind=c["revision_meta"]["condition_kind"],
            attack_type=gt["attack_class"], affected_asset=gt["affected_asset"],
            start_s=gt["start_s"], end_s=gt["end_s"],
            episode_seed=(c.get("stub_params") or {}).get("episode_seed"),
            config_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            config_path=str(p.resolve())))
    pd.DataFrame(rows).to_csv(manifest_path, index=False)


def main() -> None:
    CONF.mkdir(parents=True, exist_ok=True)
    _write(build_dev_benign(), CONF / "evidence_gate_dev_benign.csv")
    _write(build_test(), CONF / "evidence_gate_test.csv")
    note = {
        "frozen_at": "2026-08-10",
        "dev_attacks": str(DEV_ATTACKS),
        "rule": ("thresholds are selected on the dev split only and frozen in "
                 "evidence_gate_frozen.json BEFORE any run on the test set"),
        "test_manifest_sha256": hashlib.sha256(
            (CONF / "evidence_gate_test.csv").read_bytes()).hexdigest(),
    }
    (CONF / "evidence_gate_freeze.json").write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
