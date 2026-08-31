"""P0-B1: calibrate the Incident Evidence Gate on the DEVELOPMENT split only.

Method: compute passive per-tick FeatureView evidence streams (no actions
taken — the gate's inputs are observation-driven) for the 8 dev benign
configurations and the 24 dev attack configurations, then grid-search
(p_hard, p_soft, z_soft) offline on those streams.

Objective (lexicographic): (1) zero missed attacks on dev; (2) minimal benign
false-open tick fraction; (3) minimal mean attack open-latency; (4) larger
persistence preferred at ties (conservatism).

The selected parameters are FROZEN to
code/configs/ijcip_final_safeagent_20260810/evidence_gate_frozen.json before
any test-set run.

Run: python3 -m code.evaluation.final_safeagent.calibrate_gate
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...Multi_AI_Agent.telemetry_features import extract as extract_features
from ...simulation.attack_injectors import REGISTRY as ATTACKS
from ...simulation.feeder import StubFeeder

TAG = "ijcip_final_safeagent_20260810"
CONF = Path("code/configs") / TAG
OUT = Path("code/results") / TAG / "p0b_gate"

GRID_P_HARD = [1, 2, 3, 5]
GRID_P_SOFT = [3, 5, 8, 12, 20, 10_000]     # 10_000 = soft path disabled
GRID_Z_SOFT = [3.0, 3.5, 4.5, 6.0, 10.0]


def stream(cfg: dict) -> pd.DataFrame:
    """Per-tick (hard, zmax, vfrac) evidence stream, defender view, passive."""
    sp = cfg.get("stub_params") or {}
    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                        base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                        episode_seed=sp.get("episode_seed"),
                        load_sigma=float(sp.get("load_sigma", 0.08)),
                        pv_cloud_rate=float(sp.get("pv_cloud_rate", 0.004)),
                        meas_noise_kw=float(sp.get("meas_noise_kw", 0.8)))
    injectors = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
                 for a in cfg.get("attacks", [])]
    dt = float(cfg["dt_s"])
    tw, ew, rows = [], [], []
    for k in range(int(float(cfg["duration_s"]) / dt) + 1):
        t = k * dt
        for inj in injectors:
            inj.physical_mutate(t, feeder)
        feeder.solve(t)
        s, evs = feeder.read(t)
        for inj in injectors:
            s = inj.mutate_telemetry(t, s)
            evs = inj.mutate_events(t, evs)
        # NB: use the REPORTED timestamp (s.t) — a frozen/replayed sample
        # carries its capture-time t, which is what the staleness signature
        # detects. The production adapter does the same.
        tw.append({"t": s.t, "freq_hz": s.freq_hz, "v_pu": s.bus_voltages_pu,
                   "der_p_kw": s.der_p_kw, "der_p_avail": s.der_p_avail_kw,
                   "load_demand_kw": s.load_demand_kw,
                   "load_served_kw": s.load_served_kw})
        tw = tw[-61:]
        ew += [{"t": e.t, "source": e.source, "kind": e.kind,
                "payload": e.payload, "tampered": e.tampered} for e in evs]
        ew = ew[-60:]
        fv = extract_features(telemetry_window=tw, event_window=ew)
        zmax = max((abs(v) for v in fv.asset_zscores.values()), default=0.0)
        hard = bool(fv.n_tampered_events or fv.n_command_events
                    or fv.persistent_freeze or fv.n_dup_events >= 2
                    or fv.telemetry_stale_ticks >= 3)
        rows.append({"t": t, "hard": hard, "zmax": zmax,
                     "vfrac": fv.voltage_excursion_frac})
    return pd.DataFrame(rows)


def gate_open_series(df: pd.DataFrame, p_hard: int, p_soft: int,
                     z_soft: float) -> np.ndarray:
    hard = df.hard.to_numpy()
    soft = (df.zmax.to_numpy() >= z_soft) | (df.vfrac.to_numpy() > 0)
    open_ = np.zeros(len(df), dtype=bool)
    hr = sr = 0
    for i in range(len(df)):
        hr = hr + 1 if hard[i] else 0
        sr = sr + 1 if soft[i] else 0
        open_[i] = hr >= p_hard or sr >= p_soft
    return open_


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    benign = pd.read_csv(CONF / "evidence_gate_dev_benign.csv")
    attacks = pd.read_csv("code/configs/ijcip_revision_r1r2_20260805/"
                          "scenario_manifest.csv")
    attacks = attacks[attacks.attack_type != "none"]

    streams, meta = {}, {}
    for i, (_, r) in enumerate(pd.concat([benign.assign(kind="benign"),
                                          attacks.assign(kind="attack")])
                               .iterrows()):
        cfg = yaml.safe_load(Path(r.config_path).read_text())
        # Legacy dev attack configs are noise-free deterministic sims, in
        # which every telemetry payload repeats exactly and the duplicate/
        # staleness signature degenerates. For gate calibration the dev
        # attacks are run as genuine stochastic episodes with the published
        # exogenous parameters (dev-protocol choice, made before test runs).
        if r.kind == "attack" and not cfg.get("stub_params"):
            cfg["stub_params"] = dict(episode_seed=500 + i, load_sigma=0.08,
                                      pv_cloud_rate=0.004, meas_noise_kw=0.8)
        streams[r.scenario_id] = stream(cfg)
        gt = cfg.get("ground_truth") or {}
        meta[r.scenario_id] = {"kind": r.kind,
                               "start": float(gt.get("start_s") or 0),
                               "end": float(gt.get("end_s") or 0)}
        print(f"[stream] {r.scenario_id} ({r.kind})", flush=True)

    results = []
    for ph, ps, zs in itertools.product(GRID_P_HARD, GRID_P_SOFT, GRID_Z_SOFT):
        b_open_frac, latencies, misses = [], [], 0
        for sid, df in streams.items():
            o = gate_open_series(df, ph, ps, zs)
            m = meta[sid]
            if m["kind"] == "benign":
                b_open_frac.append(float(o.mean()))
            else:
                w = df.t.between(m["start"], m["end"] + 60.0).to_numpy()
                idx = np.where(o & w)[0]
                if len(idx) == 0:
                    misses += 1
                else:
                    latencies.append(float(df.t.iloc[idx[0]] - m["start"]))
        results.append({"p_hard": ph, "p_soft": ps, "z_soft": zs,
                        "dev_attack_misses": misses,
                        "benign_open_tick_frac": float(np.mean(b_open_frac)),
                        "mean_open_latency_s": (float(np.mean(latencies))
                                                if latencies else float("nan"))})
    res = pd.DataFrame(results)
    res.to_csv(OUT / "calibration_grid.csv", index=False)

    ok = res[res.dev_attack_misses == 0].copy()
    ok = ok.sort_values(["benign_open_tick_frac", "mean_open_latency_s",
                         "p_hard", "p_soft"],
                        ascending=[True, True, False, False])
    best = ok.iloc[0]
    frozen = {"p_hard": int(best.p_hard), "p_soft": int(best.p_soft),
              "z_soft": float(best.z_soft),
              "dev_attack_misses": int(best.dev_attack_misses),
              "dev_benign_open_tick_frac": float(best.benign_open_tick_frac),
              "dev_mean_open_latency_s": float(best.mean_open_latency_s),
              "frozen_at": "2026-08-10",
              "rule": "selected on dev only; frozen before any test-set run"}
    (CONF / "evidence_gate_frozen.json").write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen, indent=2))
    print(ok.head(10).to_string())


if __name__ == "__main__":
    main()
