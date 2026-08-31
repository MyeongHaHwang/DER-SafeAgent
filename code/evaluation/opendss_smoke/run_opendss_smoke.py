"""IJCIP P0 OpenDSS smoke validation (gated by env vars).

Runs the IEEE-13 command-spoof scenario through the OpenDSS-backed
``OpenDSSFeeder`` for three detectors and records the resulting
energy-impact metrics. The script is *gracefully gated*: if
``DER_OPENDSS_ENABLED`` is not set or the feeder file cannot be
loaded, it writes a single ``not_run`` row with a reason --- never
fabricates numbers.

Outputs:
    code/results/ijcip_opendss_smoke/opendss_smoke_metrics.csv
    code/results/ijcip_opendss_smoke/README.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from ...simulation.feeder import OpenDSSFeeder, StubFeeder
from ...simulation.harness import run_scenario
from ..physical_curves import physical_summary


SCENARIOS = [
    "code/simulation/scenarios/ieee13_command_spoof/config.yaml",
    "code/simulation/scenarios/ieee13_fdi_inverter/config.yaml",
]
DETECTORS = {
    "rule_ids":     "code.baselines.rule_ids.adapter:RuleIDS",
    "single_llm":   "code.baselines.single_llm.runner:SingleLLM",
    "der_secagent": "code.Multi_AI_Agent.adapter:DERSecAgentDetector",
}


def _instantiate(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def _opendss_available() -> tuple[bool, str]:
    if os.environ.get("DER_OPENDSS_ENABLED", "") not in {"1", "true", "True"}:
        return False, "DER_OPENDSS_ENABLED not set to 1"
    try:
        import opendssdirect  # noqa
    except ImportError as exc:
        return False, f"opendssdirect import failed: {exc}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="code/results/ijcip_opendss_smoke")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ok, reason = _opendss_available()
    if not ok:
        rows = []
        for s in SCENARIOS:
            for d in DETECTORS:
                rows.append({"scenario": Path(s).parent.name, "detector": d,
                              "seed": -1, "status": "not_run", "reason": reason,
                              **{k: float("nan") for k in (
                                  "ens_kwh", "curt_kwh",
                                  "voltage_min_pu", "voltage_max_pu",
                                  "voltage_violation_frac",
                                  "freq_dev_hz", "selected_action")}})
        pd.DataFrame(rows).to_csv(out / "opendss_smoke_metrics.csv", index=False)
        (out / "README.md").write_text(
            "# IJCIP P0 OpenDSS smoke validation (not run)\n\n"
            f"reason: {reason}\n\n"
            "Set `DER_OPENDSS_ENABLED=1` and ensure `opendssdirect` is\n"
            "available to run the smoke scenario.\n")
        print(f"opendss not available: {reason}; recorded {len(rows)} not_run rows")
        return

    rows = []
    for scenario_path in SCENARIOS:
        cfg = yaml.safe_load(Path(scenario_path).read_text())
        for det_name, det_spec in DETECTORS.items():
            for seed in [int(s) for s in args.seeds.split(",")]:
                detector = _instantiate(det_spec)
                # Try OpenDSSFeeder; if the feeder file fails to load,
                # fall back to StubFeeder for THIS row but mark the row
                # as "fallback_stub" rather than fabricating a real run.
                feeder = None
                row_status = "ok"; row_reason = ""
                try:
                    feeder = OpenDSSFeeder(
                        dss_path=str(Path(scenario_path).parent / cfg["feeder"]),
                        monitored_buses=cfg["monitored_buses"],
                        ders=cfg["ders"])
                except Exception as exc:
                    row_status = "fallback_stub"
                    row_reason = f"OpenDSSFeeder load failed: {exc}"[:160]
                    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                          ders=cfg["ders"])
                run_dir = run_scenario(scenario_path, detector, seed,
                                          out_root=str(Path(args.out) / "runs"),
                                          feeder=feeder)
                ts = pd.read_csv(run_dir / "timeseries.csv")
                summary = physical_summary(ts)
                v_cols = [c for c in ts.columns if c.startswith("v_pu_")]
                v_min = float(ts[v_cols].min().min()) if v_cols else float("nan")
                v_max = float(ts[v_cols].max().max()) if v_cols else float("nan")
                # selected action: read first action from decisions.jsonl
                actions = []
                dec_p = run_dir / "decisions.jsonl"
                if dec_p.exists():
                    for line in dec_p.read_text().splitlines():
                        if not line.strip(): continue
                        rec = json.loads(line)
                        actions.extend(a.get("name", "no_op")
                                          for a in rec.get("actions", []))
                first_action = next((a for a in actions if a != "no_op"), "no_op")
                rows.append({
                    "scenario": Path(scenario_path).parent.name,
                    "detector": det_name, "seed": seed,
                    "status": row_status, "reason": row_reason,
                    "ens_kwh":           summary["ens_kwh"],
                    "curt_kwh":          summary["curt_kwh"],
                    "voltage_min_pu":    v_min,
                    "voltage_max_pu":    v_max,
                    "voltage_violation_frac": summary["voltage_frac"],
                    "freq_dev_hz":       summary["freq_dev_hz"],
                    "selected_action":   first_action,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out / "opendss_smoke_metrics.csv", index=False)
    n_ok = (df["status"] == "ok").sum()
    n_fb = (df["status"] == "fallback_stub").sum()
    (out / "README.md").write_text(
        "# IJCIP P0 OpenDSS smoke validation\n\n"
        f"- ok rows         : {n_ok}\n"
        f"- fallback_stub   : {n_fb}\n\n"
        "When the OpenDSS solve fails on the configured feeder we fall\n"
        "back to the deterministic StubFeeder for that row and record\n"
        "`status='fallback_stub'`. Real OpenDSS results are only those\n"
        "with `status='ok'`.\n")
    print(f"opendss smoke: ok={n_ok} fallback_stub={n_fb}")


if __name__ == "__main__":
    main()
