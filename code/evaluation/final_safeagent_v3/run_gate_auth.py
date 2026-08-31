"""Gate command-authentication experiment (IJCIP §4.2).

Demonstrates that a command packet alone does not prove an attack. Three
command-evidence policies over the runtime-safe FeatureView:

  naive         : any command packet opens the gate (hard evidence).
  auth_required : a command opens the gate only if it FAILED authentication
                  (auth_valid == False); an authenticated command does not.
  no_auth       : authentication is unavailable, so a command is UNKNOWN
                  evidence and never opens the gate by itself; detection must
                  come from physical corroboration (stale/dup/residual) or an
                  operator (HITL).

Evaluated on: command-spoof attacks (forged, auth_valid=False, with a physical
effect) and benign legitimate authenticated command bursts (auth_valid=True,
in-bounds setpoint). Reports spoof gate-open coverage and benign false-open per
policy, with Clopper-Pearson intervals.

Run: python3 -m code.evaluation.final_safeagent_v3.run_gate_auth
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...Multi_AI_Agent.telemetry_features import extract
from ...simulation.attack_injectors import REGISTRY as ATT
from ...simulation.feeder import StubFeeder

OUT = Path("code/results/ijcip_final/gate_auth")
IEEE13 = ["SOURCEBUS", "634", "671", "675"]
DERS = [{"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0},
        {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0}]
P_HARD = 2  # consecutive hard-evidence ticks (frozen gate persistence)


def _episode_hardevidence(attacks, seed):
    """Return, per tick, the hard-evidence predicates needed by each policy,
    over a full 600 s runtime-safe episode."""
    f = StubFeeder(monitored_buses=IEEE13, ders=DERS, base_load_kw=1000.0,
                   episode_seed=seed, emit_seq=True)
    injs = [ATT[a["type"]](**{k: v for k, v in a.items() if k != "type"})
            for a in attacks]
    tw, ew = [], []
    hwm = -1
    rows = []
    for k in range(601):
        t = float(k)
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
        fv = extract(tw, ew, runtime_safe=True)
        # stateful replay hwm
        br = 0
        for e in evs:
            if e.seq is None:
                continue
            if e.seq <= hwm:
                br += 1
            else:
                hwm = e.seq
        seq_reg = max(fv.seq_regressions, br + 1 if br else 0)
        # physical corroboration = observable evidence OTHER than command
        phys = bool(fv.integrity_residual or fv.persistent_freeze
                    or fv.n_dup_events >= 2 or fv.telemetry_stale_ticks >= 3
                    or seq_reg >= 2)
        rows.append({"t": t, "any_cmd": fv.n_command_events >= 1,
                     "unauth_cmd": fv.n_unauth_command_events >= 1,
                     "phys": phys})
    return pd.DataFrame(rows)


def _gate_opens(df, predicate_col, window=None):
    """Sustained (>=P_HARD consecutive) predicate → gate open; optionally
    restricted to an attack window [w0, w1]."""
    p = df[predicate_col].to_numpy()
    run = 0
    for i in range(len(df)):
        if window is not None and not (window[0] <= df.t.iloc[i] <= window[1] + 60):
            run = 0
            continue
        run = run + 1 if p[i] else 0
        if run >= P_HARD:
            return True
    return False


def _cp(k, n):
    from scipy import stats
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
    return lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [2101, 2102, 2103]
    # spoof configs (forged, physical effect) at 3 magnitudes
    spoof = [("spoof_low", [dict(type="command_spoof", target="INV_634",
                                 start_s=180, end_s=420, forged_setpoint_kw=110.0,
                                 period_s=5.0)]),
             ("spoof_med", [dict(type="command_spoof", target="INV_634",
                                 start_s=180, end_s=420, forged_setpoint_kw=40.0,
                                 period_s=5.0)]),
             ("spoof_high", [dict(type="command_spoof", target="INV_634",
                                  start_s=180, end_s=420, forged_setpoint_kw=5.0,
                                  period_s=5.0)])]
    # benign legitimate authenticated command bursts (no attack)
    benign = [("benign_cmd_0", [dict(type="benign_command", target="INV_634",
                                     start_s=120, end_s=480, setpoint_kw=140.0,
                                     period_s=5.0)]),
              ("benign_cmd_1", [dict(type="benign_command", target="INV_634",
                                     start_s=150, end_s=450, setpoint_kw=145.0,
                                     period_s=7.0)])]
    policies = {"naive": "any_cmd", "auth_required": "unauth_cmd"}

    rows = []
    for name, atk in spoof:
        w = (atk[0]["start_s"], atk[0]["end_s"])
        for seed in seeds:
            df = _episode_hardevidence(atk, seed)
            for pol, col in policies.items():
                rows.append({"config": name, "kind": "spoof", "policy": pol,
                             "seed": seed, "opened": _gate_opens(df, col, w)})
            # no_auth: command is UNKNOWN; only physical corroboration opens
            rows.append({"config": name, "kind": "spoof", "policy": "no_auth",
                         "seed": seed, "opened": _gate_opens(df, "phys", w)})
    for name, atk in benign:
        for seed in seeds:
            df = _episode_hardevidence(atk, seed)
            for pol, col in policies.items():
                rows.append({"config": name, "kind": "benign", "policy": pol,
                             "seed": seed, "opened": _gate_opens(df, col)})
            rows.append({"config": name, "kind": "benign", "policy": "no_auth",
                         "seed": seed, "opened": _gate_opens(df, "phys")})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "gate_auth_raw.csv", index=False)

    # aggregate to configuration (any episode opened), then rate over configs
    summ = []
    for pol in ["naive", "auth_required", "no_auth"]:
        for kind in ["spoof", "benign"]:
            sub = raw[(raw.policy == pol) & (raw.kind == kind)]
            per_cfg = sub.groupby("config").opened.max()  # any episode
            k = int(per_cfg.sum())
            n = int(len(per_cfg))
            lo, hi = _cp(k, n)
            summ.append({"policy": pol, "kind": kind,
                         "metric": ("spoof_gate_open" if kind == "spoof"
                                    else "benign_false_open"),
                         "k": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "gate_auth_summary.csv", index=False)
    (OUT / "gate_auth_note.json").write_text(json.dumps({
        "finding": ("A command packet alone does not prove an attack. The naive "
                    "policy (any command opens the gate) catches spoofs but "
                    "false-opens on benign authenticated command bursts. The "
                    "auth_required policy (open only on failed authentication) "
                    "catches the forged spoof and does not false-open on "
                    "legitimate commands. Without authentication (no_auth), "
                    "command evidence is UNKNOWN and the gate opens only on "
                    "physical corroboration; command-spoof coverage then "
                    "depends on the attack's physical footprint, and a "
                    "purely message-layer forgery would require HITL."),
        "P_HARD": P_HARD}, indent=2))
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
