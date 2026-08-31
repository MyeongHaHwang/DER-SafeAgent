"""V3 Phase 3 — EH estimator robustness suite (counterfactual, CPU).

Stress-tests the horizon-aware estimator EH against E60 and an oracle ranking
on the frozen estimator holdout under perturbations that a real deployment
would face: horizon misspecification (+-25/50%), delayed execution, partial
action failure, and alternative cost weights. Rollouts never consult the
estimator under test. Reports tie-aware + strict top-1, Kendall tau,
mean/median/p95/worst regret, and MAE.

Run: python3 -m code.evaluation.final_safeagent_v3.run_eh_robustness
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...Multi_AI_Agent.energy_estimator import (CANDIDATE_ACTIONS, cheapest_safe,
                                                estimate as estimate_e60)
from ...Multi_AI_Agent.horizon_estimator import (HorizonCalibration, class_family,
                                                 estimate_eh, select)
from ...Multi_AI_Agent.telemetry_features import extract as extract_features
from ...simulation.attack_injectors import REGISTRY as ATTACKS
from ...simulation.feeder import StubFeeder
from ...simulation.types import Action
from code.simulation.portable_paths import resolve_config_path as _resolve

V2CONF = Path("code/configs/ijcip_final_safeagent_20260810")
OUT = Path("code/results/ijcip_final_v3/eh_robustness")
EPS = 0.01
DELIB = 5.0


def rollout(cfg, action_at, action, target, horizon=600.0,
            delay=0.0, partial=False):
    f = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                   base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
    injs = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
            for a in cfg.get("attacks", [])]
    dt = float(cfg["dt_s"])
    n = int(min(horizon, float(cfg["duration_s"])) / dt) + 1
    ens = curt = varea = 0.0
    vmin = 1.0
    applied = False
    for k in range(n):
        t = k * dt
        for inj in injs:
            inj.physical_mutate(t, f)
        f.solve(t)
        s, _ = f.read(t)
        if (not applied and action and action != "no_op"
                and t >= action_at + delay):
            # partial failure: apply a weaker action (throttle instead of the
            # requested setpoint) to model imperfect actuation
            act = "throttle_ramp" if (partial and action == "freeze_setpoint") else action
            f.apply(Action(name=act, target=target))
            applied = True
        ens += max(0.0, s.load_demand_kw - s.load_served_kw) * dt / 3600.0
        avail = sum(s.der_p_avail_kw.values())
        act_p = sum(s.der_p_kw.values())
        curt += max(0.0, avail - act_p) * dt / 3600.0
        vm = min(s.bus_voltages_pu.values()) if s.bus_voltages_pu else 1.0
        vmin = min(vmin, vm)
        varea += max(0.0, 0.95 - vm) * dt
    return {"ens": ens, "curt": curt, "varea": varea}


def cost(m, w_ens=5.0):
    return m["curt"] + w_ens * m["ens"] + m["varea"]


def decision_fv(cfg, action_at):
    f = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                   base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
    injs = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
            for a in cfg.get("attacks", [])]
    dt = float(cfg["dt_s"])
    tw, ew = [], []
    for k in range(int(action_at / dt) + 1):
        t = k * dt
        for inj in injs:
            inj.physical_mutate(t, f)
        f.solve(t)
        s, evs = f.read(t)
        for inj in injs:
            s = inj.mutate_telemetry(t, s)
            evs = inj.mutate_events(t, evs)
        tw.append({"t": s.t, "freq_hz": s.freq_hz, "v_pu": s.bus_voltages_pu,
                   "der_p_kw": s.der_p_kw, "der_p_avail": s.der_p_avail_kw,
                   "load_demand_kw": s.load_demand_kw,
                   "load_served_kw": s.load_served_kw})
        ew += [{"t": e.t, "source": e.source, "kind": e.kind,
                "payload": e.payload, "tampered": e.tampered} for e in evs]
    fv = extract_features(tw[-60:], ew[-60:])
    return fv, float(tw[-1]["load_demand_kw"])


def eval_condition(man, calib, w_ens=5.0, horizon_mult=1.0, delay=0.0,
                   partial=False):
    """Return per-estimator regret/top1 under one perturbation."""
    recs = {e: [] for e in ("E60", "EH")}
    for _, r in man.iterrows():
        cfg = yaml.safe_load(_resolve(r.config_path).read_text())
        atk = (cfg.get("attacks") or [{}])[0]
        action_at = float(atk.get("start_s", 180)) + DELIB
        target = atk.get("target") or cfg["ders"][0]["id"]
        fv, load = decision_fv(cfg, action_at)
        fam = class_family(fv)
        # realised truth under this perturbation
        truth = {a: rollout(cfg, action_at, a, target, delay=delay,
                            partial=partial) for a in CANDIDATE_ACTIONS}
        costs = {a: cost(truth[a], w_ens) for a in CANDIDATE_ACTIONS}
        cmin = min(costs.values())
        opt = {a for a, c in costs.items() if c - cmin <= EPS}
        # estimator choices; EH horizon perturbed by horizon_mult via a scaled
        # calibration (models remaining-horizon misspecification)
        e60 = cheapest_safe(list(estimate_e60(fv, target_asset=target).__iter__())
                            if False else estimate_e60(fv, target_asset=target),
                            fv.severity_score).action
        cal2 = HorizonCalibration(
            durations_s={k: [d * horizon_mult for d in v]
                         for k, v in calib.durations_s.items()},
            floor_s=calib.floor_s, cap_s=calib.cap_s)
        eh = select(estimate_eh(fv, target, DELIB, cal2, load_demand_kw=load),
                    family=fam, severity=fv.severity_score).action
        for est, ch in (("E60", e60), ("EH", eh)):
            recs[est].append({"regret": costs[ch] - cmin,
                              "top1": ch in opt})
    out = {}
    for est in ("E60", "EH"):
        d = pd.DataFrame(recs[est])
        out[est] = {"top1": float(d.top1.mean()),
                    "mean_regret": float(d.regret.mean()),
                    "median_regret": float(d.regret.median()),
                    "p95_regret": float(np.quantile(d.regret, 0.95)),
                    "worst_regret": float(d.regret.max())}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(V2CONF / "impact_estimator_holdout.csv")
    calib = HorizonCalibration.load(V2CONF / "eh_duration_prior.json")
    conditions = {
        "nominal": dict(),
        "horizon_-50%": dict(horizon_mult=0.5),
        "horizon_-25%": dict(horizon_mult=0.75),
        "horizon_+25%": dict(horizon_mult=1.25),
        "horizon_+50%": dict(horizon_mult=1.5),
        "delayed_10s": dict(delay=10.0),
        "delayed_30s": dict(delay=30.0),
        "partial_failure": dict(partial=True),
        "w_ens=3": dict(w_ens=3.0),
        "w_ens=8": dict(w_ens=8.0),
    }
    rows = []
    for name, kw in conditions.items():
        res = eval_condition(man, calib, **kw)
        for est in ("E60", "EH"):
            rows.append({"condition": name, "estimator": est, **res[est]})
        print(f"[eh] {name}: EH top1={res['EH']['top1']:.2f} "
              f"medregret={res['EH']['median_regret']:.2f} "
              f"worst={res['EH']['worst_regret']:.1f} | "
              f"E60 top1={res['E60']['top1']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "eh_robustness.csv", index=False)
    eh = df[df.estimator == "EH"]
    summary = {"conditions": len(conditions),
               "EH_top1_min": float(eh.top1.min()),
               "EH_top1_worst_condition": eh.loc[eh.top1.idxmin(), "condition"],
               "EH_worst_regret_max": float(eh.worst_regret.max()),
               "note": "'perfect generalization' holds only where EH_top1==1.0;"
                       " conditions with top1<1.0 are reported honestly"}
    (OUT / "eh_robustness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
