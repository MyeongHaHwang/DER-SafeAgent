"""E4: counterfactual validation of the Energy Impact estimator.

The estimator has never been checked against what actually happens. This
experiment branches the simulator at the moment of mitigation, executes *every*
valid action under the identical exogenous trajectory, and compares the
estimator's prediction with the realised physical outcome.

Ground truth is obtained by executing actions in the simulator, never by
consulting the estimator being evaluated.

Run: python3 -m code.evaluation.final_revision.run_e4_counterfactual
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...Multi_AI_Agent.energy_estimator import (CANDIDATE_ACTIONS, cheapest_safe,
                                                estimate as estimate_impacts)
from ...Multi_AI_Agent.telemetry_features import FeatureView, extract as extract_features
from ...simulation.attack_injectors import REGISTRY as ATTACKS
from ...simulation.feeder import StubFeeder
from ...simulation.types import Action

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "e4_estimator"
MANIFEST = Path("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv")


def _fv_from_dict(d: dict) -> FeatureView:
    return FeatureView(
        t=d.get("t", 0.0),
        asset_zscores={k: float(v) for k, v in (d.get("asset_zscores") or {}).items()},
        asset_p_kw={k: float(v) for k, v in (d.get("asset_p_kw") or {}).items()},
        asset_p_avail_kw={k: float(v) for k, v in (d.get("asset_p_avail_kw") or {}).items()},
        voltage_excursion_frac=float(d.get("voltage_excursion_frac", 0.0)),
        voltage_min_pu=float(d.get("voltage_min_pu", 1.0)),
        voltage_max_pu=float(d.get("voltage_max_pu", 1.0)),
        freq_dev_hz=float(d.get("freq_dev_hz", 0.0)),
        n_command_events=int(d.get("n_command_events", 0)),
        n_tampered_events=int(d.get("n_tampered_events", 0)),
        n_dup_events=int(d.get("n_dup_events", 0)),
        persistent_freeze=bool(d.get("persistent_freeze", False)),
        dominant_asset=d.get("dominant_asset"),
        dominant_signal=d.get("dominant_signal", "none"),
        severity_score=float(d.get("severity_score", 0.0)),
    )


def rollout(cfg: dict, cfg_path: str, action_at: float, action: str | None,
            target: str, horizon_s: float) -> dict:
    """Run the scenario, applying ``action`` once at ``action_at``, and return
    the realised physical metrics over the remaining horizon.

    The exogenous trajectory (attack injectors, load, PV) is identical across
    branches: only the applied action differs, which is what makes the
    comparison counterfactual rather than merely correlational.
    """
    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                        base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
    injectors = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
                 for a in cfg.get("attacks", [])]
    dt = float(cfg["dt_s"])
    n = int(min(horizon_s, float(cfg["duration_s"])) / dt) + 1
    ens = curt = 0.0
    v_area = 0.0
    v_min = 1.0
    applied = False
    recovery_t = None
    for k in range(n):
        t = k * dt
        for inj in injectors:
            inj.physical_mutate(t, feeder)
        feeder.solve(t)
        sample, _ = feeder.read(t)
        if not applied and action and action != "no_op" and t >= action_at:
            feeder.apply(Action(name=action, target=target))
            applied = True
        # realised physical accounting over this step
        served, demand = sample.load_served_kw, sample.load_demand_kw
        ens += max(0.0, demand - served) * dt / 3600.0
        avail = sum(sample.der_p_avail_kw.values())
        actual = sum(sample.der_p_kw.values())
        curt += max(0.0, avail - actual) * dt / 3600.0
        vmin = min(sample.bus_voltages_pu.values()) if sample.bus_voltages_pu else 1.0
        v_min = min(v_min, vmin)
        v_area += max(0.0, 0.95 - vmin) * dt
        if applied and recovery_t is None and vmin >= 0.95 and demand - served < 1e-6:
            recovery_t = t - action_at
    return {"ens_kwh": ens, "curt_kwh": curt, "voltage_area_pu_s": v_area,
            "voltage_min_pu": v_min,
            "recovery_s": recovery_t if recovery_t is not None else float("nan")}


def objective(m: dict, ens_weight: float = 5.0) -> float:
    """Decision cost in kWh-equivalent units.

    Dimensional note (brief §9): the deployed Coordinator adds a dimensionless
    tier penalty to kWh terms. Here every term is kWh-equivalent: curtailed
    energy, ENS weighted by ``ens_weight`` (unserved energy is worse per kWh
    than curtailed energy), and a voltage term converted at 1 kWh per pu-second
    of excursion below the ANSI band. The weights are stated so a reader can
    substitute their own.
    """
    return (m["curt_kwh"] + ens_weight * m["ens_kwh"]
            + 1.0 * m["voltage_area_pu_s"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-s", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    if args.limit:
        man = man.head(args.limit)

    rows = []
    for _, r in man.iterrows():
        cfg = yaml.safe_load(Path(r.config_path).read_text())
        atk = (cfg.get("attacks") or [{}])[0]
        action_at = float(atk.get("start_s", 180)) + 5.0
        target = atk.get("target") or (cfg["ders"][0]["id"])

        # --- state at the decision instant, for the estimator's own input ---
        feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                            base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
        injectors = [ATTACKS[a["type"]](**{k: v for k, v in a.items() if k != "type"})
                     for a in cfg.get("attacks", [])]
        tw, ew = [], []
        dt = float(cfg["dt_s"])
        for k in range(int(action_at / dt) + 1):
            t = k * dt
            for inj in injectors:
                inj.physical_mutate(t, feeder)
            feeder.solve(t)
            s, evs = feeder.read(t)
            for inj in injectors:
                s = inj.mutate_telemetry(t, s)
                evs = inj.mutate_events(t, evs)
            tw.append({"t": t, "freq_hz": s.freq_hz, "v_pu": s.bus_voltages_pu,
                       "der_p_kw": s.der_p_kw, "der_p_avail": s.der_p_avail_kw,
                       "load_demand_kw": s.load_demand_kw,
                       "load_served_kw": s.load_served_kw})
            ew += [{"t": e.t, "source": e.source, "kind": e.kind,
                    "payload": e.payload, "tampered": e.tampered} for e in evs]
        fv = extract_features(telemetry_window=tw[-60:], event_window=ew[-60:])
        preds = {e.action: e for e in estimate_impacts(fv, target_asset=target)}

        # --- counterfactual ground truth: execute every candidate ---
        truth = {a: rollout(cfg, r.config_path, action_at, a, target, args.horizon_s)
                 for a in CANDIDATE_ACTIONS}
        costs = {a: objective(m) for a, m in truth.items()}
        best = min(costs, key=costs.get)

        est_choice = cheapest_safe(list(preds.values()), fv.severity_score).action
        for a in CANDIDATE_ACTIONS:
            rows.append({
                "scenario": r.scenario_id, "configuration_hash": r.configuration_hash,
                "attack_type": r.attack_type, "feeder": ("ieee34" if r.scenario_id.startswith("rv34") else "ieee13"),
                "action": a,
                "pred_curt_kwh": preds[a].expected_curt_kwh,
                "pred_ens_kwh": preds[a].expected_ens_kwh,
                "pred_tier": preds[a].tier,
                "true_curt_kwh": truth[a]["curt_kwh"],
                "true_ens_kwh": truth[a]["ens_kwh"],
                "true_voltage_area": truth[a]["voltage_area_pu_s"],
                "true_recovery_s": truth[a]["recovery_s"],
                "true_cost": costs[a],
                "is_true_optimal": a == best,
                "estimator_choice": est_choice,
                "estimator_regret": costs[est_choice] - costs[best],
                "severity": fv.severity_score,
            })
        print(f"[cf] {r.scenario_id}: true-best={best} estimator={est_choice} "
              f"regret={costs[est_choice]-costs[best]:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "counterfactual_rollouts.csv", index=False)

    # ---- estimator quality metrics ----
    per_scen = df.groupby("scenario").first().reset_index()
    top1 = float((per_scen.estimator_choice == per_scen.apply(
        lambda r: df[(df.scenario == r.scenario) & df.is_true_optimal].action.iloc[0],
        axis=1)).mean())
    mae_curt = float((df.pred_curt_kwh - df.true_curt_kwh).abs().mean())
    mae_ens = float((df.pred_ens_kwh - df.true_ens_kwh).abs().mean())
    rng = np.ptp(df.true_curt_kwh) or 1.0
    nrmse_curt = float(np.sqrt(((df.pred_curt_kwh - df.true_curt_kwh) ** 2).mean()) / rng)
    def _spearman(a, b):
        ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
        return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")
    rho_curt = _spearman(df.pred_curt_kwh, df.true_curt_kwh)
    rho_ens = _spearman(df.pred_ens_kwh, df.true_ens_kwh)
    regret = per_scen.estimator_regret
    metrics = {
        "n_scenarios": int(per_scen.shape[0]),
        "n_counterfactual_rollouts": int(len(df)),
        "top1_optimal_action_accuracy": top1,
        "mae_curtailment_kwh": mae_curt, "mae_ens_kwh": mae_ens,
        "nrmse_curtailment": nrmse_curt,
        "spearman_curtailment": rho_curt, "spearman_ens": rho_ens,
        "median_regret": float(regret.median()),
        "p95_regret": float(np.quantile(regret, 0.95)),
        "max_regret": float(regret.max()),
        "frac_scenarios_zero_regret": float((regret <= 1e-9).mean()),
        "objective_note": ("all terms kWh-equivalent: curt + 5*ENS + "
                           "1.0*voltage_area(pu*s); weights stated, not fitted"),
    }
    (OUT / "estimator_validation.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame([metrics]).to_csv(OUT / "estimator_validation.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
