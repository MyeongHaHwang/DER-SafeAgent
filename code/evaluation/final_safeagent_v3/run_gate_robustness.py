"""V3 Phase 1 — runtime-safe Evidence Gate robustness.

Re-calibrates the gate on the DEVELOPMENT split under runtime-safe observables
(no `tampered` oracle; sequence numbers emitted), freezes thresholds, and
evaluates on the frozen 32-config gate test set plus a per-magnitude FDI
breakdown. Reports gate-open coverage split by family, benign false-action
(gate-open on benign), and time-to-open, with exact counts and Clopper-Pearson
95% intervals.

Thresholds are chosen on dev only and hash-frozen before the test evaluation.

Run:
  python3 -m code.evaluation.final_safeagent_v3.run_gate_robustness --calibrate
  python3 -m code.evaluation.final_safeagent_v3.run_gate_robustness --evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...Multi_AI_Agent.evidence_gate import EvidenceGate, GateParams
from ...Multi_AI_Agent.telemetry_features import extract
from ...simulation.attack_injectors import REGISTRY as ATT
from ...simulation.feeder import StubFeeder
from code.simulation.portable_paths import resolve_config_path as _resolve

V2CONF = Path("code/configs/ijcip_final_safeagent_20260810")
CONF = Path("code/configs/ijcip_final_v3")
OUT = Path("code/results/ijcip_final_v3/gate_robustness")
DEV_ATTACKS = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")

GRID_P_HARD = [2, 3, 5]
GRID_P_SOFT = [5, 8, 12, 10_000]   # 10_000 = soft path disabled
GRID_Z_SOFT = [3.0, 4.0, 6.0]


def _run_gate(cfg, params: GateParams, seed, runtime_safe=True):
    """Step a stateful gate over a full episode; return (opened_in_window,
    t_open, opened_benign_any, family)."""
    sp = cfg.get("stub_params") or {}
    f = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                   base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                   episode_seed=(sp.get("episode_seed", seed)),
                   load_sigma=float(sp.get("load_sigma", 0.08)),
                   pv_cloud_rate=float(sp.get("pv_cloud_rate", 0.004)),
                   meas_noise_kw=float(sp.get("meas_noise_kw", 0.8)),
                   emit_seq=True)
    injs = [ATT[a["type"]](**{k: v for k, v in a.items() if k != "type"})
            for a in cfg.get("attacks", [])]
    gt = cfg.get("ground_truth") or {}
    s0, s1 = float(gt.get("start_s") or 0), float(gt.get("end_s") or 0)
    gate = EvidenceGate(params)
    tw, ew = [], []
    hwm = -1
    opened_win, t_open, opened_any = False, None, False
    dur = int(float(cfg["duration_s"]) / float(cfg["dt_s"]))
    for k in range(dur + 1):
        t = k * float(cfg["dt_s"])
        for j in injs:
            j.physical_mutate(t, f)
        f.solve(t)
        s, evs = f.read(t)
        for j in injs:
            s = j.mutate_telemetry(t, s)
            evs = j.mutate_events(t, evs)
        tw.append({"t": s.t, "freq_hz": s.freq_hz, "v_pu": s.bus_voltages_pu,
                   "der_p_kw": s.der_p_kw, "der_p_avail": s.der_p_avail_kw,
                   "load_demand_kw": s.load_demand_kw,
                   "load_served_kw": s.load_served_kw})
        tw = tw[-61:]
        ew += [{"t": e.t, "source": e.source, "kind": e.kind,
                "payload": e.payload, "tampered": e.tampered, "seq": e.seq}
               for e in evs]
        ew = ew[-60:]
        fv = extract(tw, ew, runtime_safe=runtime_safe)
        if runtime_safe:
            br = 0
            for e in evs:
                if e.seq is None:
                    continue
                if e.seq <= hwm:
                    br += 1
                else:
                    hwm = e.seq
            if br:
                fv.seq_regressions = max(fv.seq_regressions, br + 1)
        opened = gate.update(fv)
        if opened:
            opened_any = True
            if s0 and s0 <= t <= s1 + 60.0 and not opened_win:
                opened_win, t_open = True, t - s0
    return opened_win, t_open, opened_any, (gt.get("attack_class") or "none")


def _fdi_mag_configs():
    """Per-magnitude FDI configs (low/med/high) on IEEE-13 INV_634."""
    base = dict(name="", description="fdi mag probe",
                feeder="../../ieee13_fdi_inverter/feeder.dss", dt_s=1.0,
                duration_s=600,
                monitored_buses=["SOURCEBUS", "634", "671", "675"],
                ders=[{"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0},
                      {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0}],
                base_load_kw=1000.0,
                stub_params=dict(episode_seed=901, meas_noise_kw=0.8))
    out = []
    for mag, bias in (("low", 20.0), ("med", 60.0), ("high", 120.0)):
        c = dict(base)
        c["name"] = f"fdi_{mag}"
        c["attacks"] = [dict(type="fdi", target="INV_634", start_s=180,
                             end_s=420, bias_kw=bias)]
        c["ground_truth"] = dict(attack_class="fdi", affected_asset="INV_634",
                                 start_s=180, end_s=420)
        c["_mag"] = mag
        out.append(c)
    return out


def calibrate():
    CONF.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    benign = pd.read_csv(V2CONF / "evidence_gate_dev_benign.csv")
    attacks = pd.read_csv(DEV_ATTACKS)
    attacks = attacks[attacks.attack_type != "none"]
    dev = []
    for _, r in benign.iterrows():
        dev.append(("benign", yaml.safe_load(_resolve(r.config_path).read_text())))
    for i, (_, r) in enumerate(attacks.iterrows()):
        cfg = yaml.safe_load(_resolve(r.config_path).read_text())
        if not cfg.get("stub_params"):
            cfg["stub_params"] = dict(episode_seed=500 + i, meas_noise_kw=0.8)
        dev.append(("attack", cfg))

    rows = []
    for ph, ps, zs in itertools.product(GRID_P_HARD, GRID_P_SOFT, GRID_Z_SOFT):
        params = GateParams(p_hard=ph, p_soft=ps, z_soft=zs)
        misses, b_open, lat = 0, 0, []
        nb = na = 0
        for kind, cfg in dev:
            ow, to, oa, _ = _run_gate(cfg, params, seed=0, runtime_safe=True)
            if kind == "benign":
                nb += 1
                b_open += int(oa)
            else:
                na += 1
                if ow:
                    lat.append(to)
                else:
                    misses += 1
        rows.append({"p_hard": ph, "p_soft": ps, "z_soft": zs,
                     "dev_attack_misses": misses, "n_attack": na,
                     "benign_open_frac": b_open / max(nb, 1),
                     "mean_open_latency_s": float(np.mean(lat)) if lat else float("nan")})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "calibration_grid.csv", index=False)
    # Design decision (documented): the gate's purpose is benign specificity,
    # so we select the HARD-EVIDENCE-ONLY operating point (soft path disabled,
    # p_soft huge) that maximises benign specificity. Without the removed
    # `tampered` oracle this cannot also catch sub-capacity FDI (which has no
    # observable hard signature) --- that miss is reported as an explicit
    # limitation rather than bought back with a benign-flooding soft path
    # (the soft-enabled Pareto point is retained in calibration_grid.csv).
    hard_only = grid[grid.p_soft >= 10_000].sort_values(
        ["benign_open_frac", "dev_attack_misses", "p_hard"],
        ascending=[True, True, True]).iloc[0]
    ok = hard_only
    soft_best = grid.sort_values(["dev_attack_misses", "benign_open_frac"],
                                 ascending=[True, True]).iloc[0]
    frozen = {"p_hard": int(ok.p_hard), "p_soft": int(ok.p_soft),
              "z_soft": float(ok.z_soft), "runtime_safe": True,
              "dev_attack_misses_hard_only": int(ok.dev_attack_misses),
              "dev_benign_open_frac": float(ok.benign_open_frac),
              "soft_enabled_alt": {"p_hard": int(soft_best.p_hard),
                                   "p_soft": int(soft_best.p_soft),
                                   "z_soft": float(soft_best.z_soft),
                                   "dev_misses": int(soft_best.dev_attack_misses),
                                   "dev_benign_open_frac": float(soft_best.benign_open_frac)},
              "frozen_at": "2026-08-11",
              "rule": "runtime-safe HARD-ONLY gate; calibrated on dev only; "
                      "frozen before test; soft-enabled alternative recorded "
                      "for the honest coverage/false-alarm tradeoff"}
    p = CONF / "evidence_gate_v3_frozen.json"
    p.write_text(json.dumps(frozen, indent=2))
    frozen["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text(json.dumps(frozen, indent=2))
    print(json.dumps(frozen, indent=2))


def _cp_interval(k, n):
    """Clopper-Pearson 95% CI for k/n."""
    from scipy import stats
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((CONF / "evidence_gate_v3_frozen.json").read_text())
    params = GateParams(p_hard=frozen["p_hard"], p_soft=frozen["p_soft"],
                        z_soft=frozen["z_soft"])
    test = pd.read_csv(V2CONF / "evidence_gate_test.csv")

    rows = []
    for _, r in test.iterrows():
        cfg = yaml.safe_load(_resolve(r.config_path).read_text())
        ow, to, oa, fam = _run_gate(cfg, params, seed=0, runtime_safe=True)
        rows.append({"scenario": r.scenario_id, "kind": r.condition_kind,
                     "attack_type": r.attack_type, "opened_in_window": ow,
                     "opened_any": oa, "t_open_s": to})
    # per-magnitude FDI
    for c in _fdi_mag_configs():
        ow, to, oa, fam = _run_gate(c, params, seed=0, runtime_safe=True)
        rows.append({"scenario": c["name"], "kind": "attack",
                     "attack_type": f"fdi_{c['_mag']}", "opened_in_window": ow,
                     "opened_any": oa, "t_open_s": to})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gate_v3_raw.csv", index=False)

    # summaries
    summ = []
    benign = df[df.kind.str.startswith("benign")]
    k = int(benign.opened_any.sum())
    n = len(benign)
    lo, hi = _cp_interval(k, n)
    summ.append({"group": "benign_false_open", "k": k, "n": n,
                 "rate": k / max(n, 1), "ci_lo": lo, "ci_hi": hi})
    for fam in sorted(df[df.kind == "attack"].attack_type.unique()):
        sub = df[(df.kind == "attack") & (df.attack_type == fam)]
        k = int(sub.opened_in_window.sum())
        n = len(sub)
        lo, hi = _cp_interval(k, n)
        summ.append({"group": f"gateopen_{fam}", "k": k, "n": n,
                     "rate": k / max(n, 1), "ci_lo": lo, "ci_hi": hi,
                     "mean_t_open_s": float(sub.t_open_s.dropna().mean())
                     if sub.t_open_s.notna().any() else float("nan")})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "gate_v3_summary.csv", index=False)
    print(s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    if args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
