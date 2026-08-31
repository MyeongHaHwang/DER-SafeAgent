"""P0-A: counterfactual validation of E60 / EH / EMH on dev + frozen holdout.

Protocol (identical to E4, extended):
- decision instant t_m = attack start + 5 s (first-trigger + deliberation);
- FeatureView computed from the *tampered* decision-time windows (defender view);
- counterfactual ground truth: branch the simulator at t_m, execute every
  candidate under the identical exogenous trajectory to the episode end,
  record realised curtailment / ENS / voltage area / scalarised cost
  (curt + 5*ENS + 1*voltage_area). The rollout never consults any estimator.
- E60 selects via the legacy ``cheapest_safe``; EH/EMH via horizon-aware
  ``select``. The oracle is argmin of realised cost.
- Top-1 uses epsilon-tie handling (optimal iff realised regret <= 0.01
  kWh-eq); the legacy strict-argmax top-1 is also reported.

Outputs (explicit --out, never a legacy results directory):
  code/results/ijcip_final_safeagent_20260810/p0a_estimator/
    rollouts_<split>.csv     per (config x action) predictions + ground truth
    selections_<split>.csv   per (config x estimator) choice + regret
    metrics_<split>.json     metric suite with bootstrap 95% CIs

Run: python3 -m code.evaluation.final_safeagent.run_p0a_estimator --split dev
     python3 -m code.evaluation.final_safeagent.run_p0a_estimator --split holdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...Multi_AI_Agent.energy_estimator import (CANDIDATE_ACTIONS, cheapest_safe,
                                                estimate as estimate_e60)
from ...Multi_AI_Agent.horizon_estimator import (HorizonCalibration, class_family,
                                                 estimate_eh, estimate_emh, select)
from ...Multi_AI_Agent.telemetry_features import extract as extract_features
from ...simulation.attack_injectors import REGISTRY as ATTACKS
from ...simulation.feeder import StubFeeder
from ...simulation.types import Action

TAG = "ijcip_final_safeagent_20260810"
OUT = Path("code/results") / TAG / "p0a_estimator"
CONF = Path("code/configs") / TAG
EPS_TIE = 0.01           # kWh-eq: actions within EPS of the minimum are optimal
DELIBERATION_S = 5.0


def rollout(cfg: dict, action_at: float, action: str | None, target: str,
            horizon_s: float = 600.0) -> dict:
    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                        base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
    injectors = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
                 for a in cfg.get("attacks", [])]
    dt = float(cfg["dt_s"])
    n = int(min(horizon_s, float(cfg["duration_s"])) / dt) + 1
    ens = curt = v_area = 0.0
    v_min, applied = 1.0, False
    for k in range(n):
        t = k * dt
        for inj in injectors:
            inj.physical_mutate(t, feeder)
        feeder.solve(t)
        sample, _ = feeder.read(t)
        if not applied and action and action != "no_op" and t >= action_at:
            feeder.apply(Action(name=action, target=target))
            applied = True
        ens += max(0.0, sample.load_demand_kw - sample.load_served_kw) * dt / 3600.0
        avail = sum(sample.der_p_avail_kw.values())
        actual = sum(sample.der_p_kw.values())
        curt += max(0.0, avail - actual) * dt / 3600.0
        vmin = min(sample.bus_voltages_pu.values()) if sample.bus_voltages_pu else 1.0
        v_min = min(v_min, vmin)
        v_area += max(0.0, 0.95 - vmin) * dt
    return {"ens_kwh": ens, "curt_kwh": curt, "voltage_area_pu_s": v_area,
            "voltage_min_pu": v_min}


def objective(m: dict, ens_weight: float = 5.0) -> float:
    return m["curt_kwh"] + ens_weight * m["ens_kwh"] + m["voltage_area_pu_s"]


def decision_state(cfg: dict, action_at: float):
    """Defender-visible FeatureView + load at the decision instant."""
    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                        base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
    injectors = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
                 for a in cfg.get("attacks", [])]
    dt = float(cfg["dt_s"])
    tw, ew = [], []
    for k in range(int(action_at / dt) + 1):
        t = k * dt
        for inj in injectors:
            inj.physical_mutate(t, feeder)
        feeder.solve(t)
        s, evs = feeder.read(t)
        for inj in injectors:
            s = inj.mutate_telemetry(t, s)
            evs = inj.mutate_events(t, evs)
        # reported timestamp (s.t): frozen/replayed samples keep capture time
        tw.append({"t": s.t, "freq_hz": s.freq_hz, "v_pu": s.bus_voltages_pu,
                   "der_p_kw": s.der_p_kw, "der_p_avail": s.der_p_avail_kw,
                   "load_demand_kw": s.load_demand_kw,
                   "load_served_kw": s.load_served_kw})
        ew += [{"t": e.t, "source": e.source, "kind": e.kind,
                "payload": e.payload, "tampered": e.tampered} for e in evs]
    fv = extract_features(telemetry_window=tw[-60:], event_window=ew[-60:])
    load_kw = float(tw[-1]["load_demand_kw"])
    return fv, load_kw


def _rank_corr(pred: list[float], true: list[float]) -> tuple[float, float]:
    """(Spearman rho, Kendall tau) for one configuration's action ranking."""
    pr = pd.Series(pred).rank().to_numpy()
    tr = pd.Series(true).rank().to_numpy()
    rho = float(np.corrcoef(pr, tr)[0, 1]) if len(set(pr)) > 1 and len(set(tr)) > 1 else float("nan")
    n = len(pred)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (pred[i] - pred[j]) * (true[i] - true[j])
            conc += s > 0
            disc += s < 0
    denom = conc + disc
    tau = float((conc - disc) / denom) if denom else float("nan")
    return rho, tau


def _boot_ci(x: np.ndarray, stat=np.mean, n=5000, seed=13) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return float("nan"), float("nan")
    vals = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "holdout"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(CONF / f"impact_estimator_{args.split}.csv")
    calib = HorizonCalibration.load(CONF / "eh_duration_prior.json")

    roll_rows, sel_rows = [], []
    for _, r in man.iterrows():
        cfg = yaml.safe_load(Path(r.config_path).read_text())
        atk = (cfg.get("attacks") or [{}])[0]
        action_at = float(atk.get("start_s", 180)) + DELIBERATION_S
        target = atk.get("target") or cfg["ders"][0]["id"]
        scen = cfg["name"]

        fv, load_kw = decision_state(cfg, action_at)
        fam = class_family(fv)
        elapsed = DELIBERATION_S

        preds = {
            "E60": {e.action: e for e in estimate_e60(fv, target_asset=target)},
            "EH": {e.action: e for e in estimate_eh(fv, target, elapsed, calib,
                                                    load_demand_kw=load_kw)},
            "EMH": {e.action: e for e in estimate_emh(fv, target, elapsed, calib,
                                                      load_demand_kw=load_kw)},
        }
        truth = {a: rollout(cfg, action_at, a, target) for a in CANDIDATE_ACTIONS}
        costs = {a: objective(m) for a, m in truth.items()}
        cmin = min(costs.values())
        optimal_set = {a for a, c in costs.items() if c - cmin <= EPS_TIE}
        strict_best = min(costs, key=costs.get)

        choices = {
            "E60": cheapest_safe(list(preds["E60"].values()), fv.severity_score).action,
            "EH": select(list(preds["EH"].values()), family=fam,
                         severity=fv.severity_score).action,
            "EMH": select(list(preds["EMH"].values()), family=fam,
                          severity=fv.severity_score).action,
        }

        for a in CANDIDATE_ACTIONS:
            row = {"scenario": scen, "split": args.split,
                   "attack_type": r.attack_type, "action": a,
                   "class_family_at_decision": fam,
                   "true_curt_kwh": truth[a]["curt_kwh"],
                   "true_ens_kwh": truth[a]["ens_kwh"],
                   "true_voltage_area": truth[a]["voltage_area_pu_s"],
                   "true_cost": costs[a],
                   "in_optimal_set": a in optimal_set,
                   "is_strict_argmin": a == strict_best}
            for est in ("E60", "EH", "EMH"):
                row[f"{est}_pred_curt_kwh"] = preds[est][a].expected_curt_kwh
                row[f"{est}_pred_ens_kwh"] = preds[est][a].expected_ens_kwh
            roll_rows.append(row)

        for est, choice in choices.items():
            pc = [preds[est][a].expected_curt_kwh + 5.0 * preds[est][a].expected_ens_kwh
                  for a in CANDIDATE_ACTIONS]
            tc = [costs[a] for a in CANDIDATE_ACTIONS]
            rho, tau = _rank_corr(pc, tc)
            pred_order = sorted(CANDIDATE_ACTIONS,
                                key=lambda a: preds[est][a].expected_curt_kwh
                                + 5.0 * preds[est][a].expected_ens_kwh)
            sel_rows.append({
                "scenario": scen, "split": args.split, "estimator": est,
                "attack_type": r.attack_type,
                "class_family_at_decision": fam,
                "choice": choice, "true_best_strict": strict_best,
                "optimal_set": ";".join(sorted(optimal_set)),
                "top1_eps": choice in optimal_set,
                "top1_strict": choice == strict_best,
                "top2_recall": strict_best in pred_order[:2],
                "regret": costs[choice] - cmin,
                "spearman": rho, "kendall_tau": tau,
                "realised_ens_kwh": truth[choice]["ens_kwh"],
                "realised_curt_kwh": truth[choice]["curt_kwh"],
            })
        print(f"[p0a:{args.split}] {scen} fam={fam} "
              + " ".join(f"{e}={choices[e]}({costs[choices[e]]-cmin:.2f})"
                         for e in choices), flush=True)

    roll = pd.DataFrame(roll_rows)
    sel = pd.DataFrame(sel_rows)
    roll.to_csv(OUT / f"rollouts_{args.split}.csv", index=False)
    sel.to_csv(OUT / f"selections_{args.split}.csv", index=False)

    metrics = {}
    for est in ("E60", "EH", "EMH"):
        s = sel[sel.estimator == est]
        reg = s.regret.to_numpy()
        mae_c = float((roll[f"{est}_pred_curt_kwh"] - roll.true_curt_kwh).abs().mean())
        mae_e = float((roll[f"{est}_pred_ens_kwh"] - roll.true_ens_kwh).abs().mean())
        lo_t, hi_t = _boot_ci(s.top1_eps.to_numpy().astype(float))
        lo_r, hi_r = _boot_ci(reg, stat=np.median)
        metrics[est] = {
            "n_configurations": int(len(s)),
            "top1_eps": float(s.top1_eps.mean()),
            "top1_eps_ci95": [lo_t, hi_t],
            "top1_strict_legacy": float(s.top1_strict.mean()),
            "top2_recall": float(s.top2_recall.mean()),
            "spearman_mean": float(s.spearman.mean(skipna=True)),
            "kendall_tau_mean": float(s.kendall_tau.mean(skipna=True)),
            "mae_curtailment_kwh": mae_c, "mae_ens_kwh": mae_e,
            "mean_regret": float(reg.mean()),
            "median_regret": float(np.median(reg)),
            "median_regret_ci95": [lo_r, hi_r],
            "p95_regret": float(np.quantile(reg, 0.95)),
            "zero_regret_frac": float((reg <= EPS_TIE).mean()),
            "realised_ens_kwh_mean": float(s.realised_ens_kwh.mean()),
            "realised_curt_kwh_mean": float(s.realised_curt_kwh.mean()),
        }
    meta = {"split": args.split, "n_configurations": int(man.shape[0]),
            "eps_tie_kwh": EPS_TIE, "estimators": metrics,
            "note": ("prior frozen from dev before holdout evaluation; "
                     "rollouts never consult any estimator")}
    (OUT / f"metrics_{args.split}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
