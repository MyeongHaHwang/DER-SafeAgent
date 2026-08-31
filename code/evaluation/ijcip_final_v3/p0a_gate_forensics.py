"""P0-A forensic audit: resolve the cross-arm Evidence-Gate coverage discrepancy.

The holdout table reports different `gate_open_covered` per arm (D0 0.949,
QPROJ/Q1/L1 0.657, LPROJ 0.495, OPROJ 0.960). The gate is deterministic and
model-independent and runs BEFORE consultation, so this needs explaining.

Root-cause hypothesis (from run_holdout_e2e.py:107-109): `gate_open_covered`
reads `detection.attack_class != "none"` from the CLOSED-LOOP decisions.jsonl.
`attack_class` is non-none only when the gate is open AND the inner detector
runs; once an arm acts, telemetry changes and the gate can close/never-open, so
the metric is a closed-loop, arm-dependent quantity, not the pre-action gate
recall.

This script separates, per (scenario, seed, arm), from the preserved per-tick
logs:
  - initial_detector_gate_recall : gate open at any tick in the attack window
                                   BEFORE that arm's first action (pre-action).
  - closed_loop_gate_coverage    : the current metric (gate open at any tick).
  - time_to_gate_open            : first gate-open tick minus attack start.
  - gate_event_count             : number of gate-open ticks.
And it HARD-ASSERTS deterministic gate invariance: for each (scenario, seed),
across all gated arms, the gate-open decision at every tick strictly before the
earliest first-action tick of any arm must be identical (identical pre-action
observation => identical deterministic gate output).

Run: python3 -m code.evaluation.ijcip_final_v3.p0a_gate_forensics
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

RUNS = Path("code/results/ijcip_final_v3/holdout_e2e/runs")
OUT = Path("code/results/ijcip_final_v3/p0_forensics")
MAN = Path("code/configs/ijcip_final_v3/holdout_v3_manifest.csv")
# Runtime-safe gated arms: identical frozen gate on observable-only FeatureView.
# OPROJ is the ORACLE upper bound (its detector reads ground-truth class), so it
# is excluded from the deterministic-gate invariance test by design, not by bug.
RTSAFE_GATED = ["D0", "Q1", "L1", "QPROJ", "LPROJ"]
GATED = RTSAFE_GATED + ["OPROJ"]
ALL = GATED + ["bareQ", "bareL"]
_WIN = {}  # scenario -> (s0, s1) attack window (+60s tail), populated in main()


def _load(arm, scen, seed):
    f = RUNS / arm / scen / arm / f"seed{seed}" / "decisions.jsonl"
    if not f.exists():
        return None
    rows = []
    for line in f.read_text().splitlines():
        d = json.loads(line)
        det = d.get("detection") or {}
        rows.append((float(d["t"]),
                     det.get("attack_class", "none") != "none",
                     bool(d.get("actions"))))
    return rows  # list of (t, gate_open_proxy, acted)


def _attack_window(scen):
    """Attack window (+60 s tail) from the frozen holdout manifest."""
    return _WIN.get(scen)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MAN)
    for _, r in man.iterrows():
        if str(r.kind) == "attack" and float(r.start_s) > 0:
            _WIN[r.scenario_id] = (float(r.start_s), float(r.end_s) + 60.0)
    # discover scenarios from D0 runs (all arms share scenario set)
    scens = sorted({p.name for p in (RUNS / "D0").iterdir() if p.is_dir()})
    rows = []
    invariance_checks = 0
    invariance_violations = []
    for scen in scens:
        win = _attack_window(scen)
        is_attack = win is not None and win[0] > 0
        for seed in (0, 1, 2):
            traces = {a: _load(a, scen, seed) for a in ALL}
            traces = {a: t for a, t in traces.items() if t}
            if not traces:
                continue
            # earliest first-action tick across gated arms (pre-action horizon)
            first_act = {}
            for a, tr in traces.items():
                fa = next((t for (t, g, acted) in tr if acted), None)
                first_act[a] = fa
            gated_present = [a for a in RTSAFE_GATED if a in traces]
            horizon = min([first_act[a] for a in gated_present
                           if first_act[a] is not None], default=None)
            # --- deterministic gate invariance test (pre-action) ---
            if is_attack and gated_present:
                # compare gate-open decision tick-by-tick before `horizon`
                ref = traces[gated_present[0]]
                ref_map = {t: g for (t, g, _) in ref}
                for a in gated_present[1:]:
                    for (t, g, _) in traces[a]:
                        if horizon is not None and t >= horizon:
                            continue
                        if win[0] <= t <= win[1]:
                            invariance_checks += 1
                            if ref_map.get(t) != g:
                                invariance_violations.append(
                                    (scen, seed, a, t, ref_map.get(t), g))
            # --- per-arm separated metrics ---
            for a, tr in traces.items():
                if not is_attack:
                    continue
                s0, s1 = win
                fa = first_act[a]
                open_ticks = [t for (t, g, _) in tr if g and s0 <= t <= s1]
                pre_open = [t for (t, g, _) in tr if g and s0 <= t <= s1
                            and (fa is None or t < fa)]
                rows.append({
                    "scenario": scen, "seed": seed, "arm": a,
                    "closed_loop_gate_coverage": len(open_ticks) > 0,
                    "initial_detector_gate_recall": len(pre_open) > 0,
                    "time_to_gate_open": (min(open_ticks) - s0)
                                          if open_ticks else None,
                    "gate_event_count": len(open_ticks),
                    "first_action_s": fa,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p0a_gate_trace_metrics.csv", index=False)

    # per-arm summary (attack configs; average within config first)
    summ = (df.groupby(["arm", "scenario"])
              .agg(closed=("closed_loop_gate_coverage", "mean"),
                   initial=("initial_detector_gate_recall", "mean"),
                   ttg=("time_to_gate_open", "mean"),
                   nopen=("gate_event_count", "mean"))
              .reset_index()
              .groupby("arm").agg(closed_loop_gate_coverage=("closed", "mean"),
                                  initial_detector_gate_recall=("initial", "mean"),
                                  time_to_gate_open_s=("ttg", "mean"),
                                  gate_event_count=("nopen", "mean"),
                                  n_config=("scenario", "nunique"))
              .reset_index())
    summ.to_csv(OUT / "p0a_gate_summary.csv", index=False)

    result = {
        "invariance_test": {
            "description": ("For each (scenario,seed), across gated arms, the "
                            "gate-open decision at every attack-window tick "
                            "strictly before the earliest first-action tick "
                            "must be identical (deterministic, model-independent "
                            "gate on identical pre-action observation)."),
            "ticks_compared": invariance_checks,
            "violations": len(invariance_violations),
            "PASS": len(invariance_violations) == 0,
            "sample_violations": invariance_violations[:10],
        },
        "per_arm_summary": summ.round(3).to_dict(orient="records"),
    }
    (OUT / "p0a_result.json").write_text(json.dumps(result, indent=2, default=str))
    print("=== P0-A: deterministic gate invariance (pre-action) ===")
    print(f"ticks compared: {invariance_checks}  violations: "
          f"{len(invariance_violations)}  PASS: {result['invariance_test']['PASS']}")
    print()
    print("=== P0-A: separated per-arm gate metrics (attack configs) ===")
    print(summ.round(3).to_string(index=False))
    if invariance_violations:
        print("\nFIRST VIOLATIONS:", invariance_violations[:5])


if __name__ == "__main__":
    main()
