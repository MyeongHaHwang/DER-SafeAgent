"""Co-simulation harness.

Time-stepped loop:
  for t in steps:
    feeder.solve(t)
    sample = read_telemetry(feeder)
    events = read_events(feeder)
    for inj in attack_injectors: sample, events = inj.mutate(...)
    detection, actions = detector.step(t, sample, events)
    for a in actions: feeder.apply(mitigations[a.name])
    log(sample, events, detection, actions)

Outputs (per run):
  runs/<scenario>/<detector>/<seed>/timeseries.csv
  runs/<scenario>/<detector>/<seed>/decisions.jsonl
  runs/<scenario>/<detector>/<seed>/manifest.json

OpenDSS access is wrapped behind `FeederAdapter` (feeder.py) so unit tests can
substitute a stub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .attack_injectors import REGISTRY as ATTACK_REGISTRY
from .feeder import FeederAdapter, OpenDSSFeeder
from .types import Action, Detection, Detector, EventLog, TelemetrySample


def _config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _telemetry_to_row(t: float, s: TelemetrySample) -> dict:
    row = {"t": t, "freq": s.freq_hz, "load_demand": s.load_demand_kw,
           "load_served": s.load_served_kw}
    for bus, v in s.bus_voltages_pu.items():
        row[f"v_pu_{bus}"] = v
    for asset, p in s.der_p_kw.items():
        row[f"p_pv_{asset}" if asset.startswith("INV") else f"p_bess_{asset}"] = p
    for asset, p in s.der_p_avail_kw.items():
        row[f"p_pv_avail_{asset}"] = p
    return row


def run_scenario(
    config_path: str,
    detector: Detector,
    seed: int,
    out_root: str = "runs",
    feeder: FeederAdapter | None = None,
    operator=None,                       # HITLOperator | None
    extra_manifest: dict | None = None,  # e.g. LLM backend provenance
) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    np.random.seed(seed)

    feeder = feeder or OpenDSSFeeder(
        dss_path=str(Path(config_path).parent / cfg["feeder"]),
        monitored_buses=cfg["monitored_buses"],
        ders=cfg["ders"],
    )
    injectors = [ATTACK_REGISTRY[a["type"]](**{k: v for k, v in a.items() if k != "type"})
                 for a in cfg.get("attacks", [])]

    out = Path(out_root) / cfg["name"] / detector.name / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    decisions: list[dict] = []
    hitl_log: list[dict] = []
    step_latencies_ms: list[float] = []
    dt = cfg["dt_s"]
    duration = cfg["duration_s"]

    for k in range(int(duration / dt) + 1):
        t = k * dt
        for inj in injectors:
            inj.physical_mutate(t, feeder)
        feeder.solve(t)
        sample, events = feeder.read(t)
        for inj in injectors:
            sample = inj.mutate_telemetry(t, sample)
            events = inj.mutate_events(t, events)

        _t0 = time.perf_counter()
        det, actions = detector.step(t, sample, events)
        step_latencies_ms.append((time.perf_counter() - _t0) * 1000.0)

        # HITL routing: if the detector exposes its last cautioned decision and
        # an operator is wired in, queue any flagged actions for approval.
        if operator is not None and getattr(detector, "_last_decision", None):
            last = detector._last_decision  # type: ignore[attr-defined]
            if last.get("caution") and actions == [] and last.get("proposed_action", "no_op") != "no_op":
                from .hitl import PendingApproval
                operator.submit(t, PendingApproval(
                    decision_id=last.get("decision_id", f"d-{t}"),
                    submitted_at=t,
                    proposed_action=Action(name=last.get("proposed_action", "no_op"),
                                          target=last.get("proposed_action_target") or ""),
                    affected_asset=last.get("affected_asset") or "",
                    rationale=last.get("rationale", ""),
                ))

        # Resolve any approvals whose response delay or SLO has elapsed.
        if operator is not None:
            for outcome in operator.step(t):
                hitl_log.append({"t": t, "decision_id": outcome.decision_id,
                                 "verdict": outcome.verdict, "note": outcome.note,
                                 "actions": [a.__dict__ for a in outcome.actions]})
                actions = list(actions) + outcome.actions

        for a in actions:
            feeder.apply(a)

        rows.append(_telemetry_to_row(t, sample))
        rec = {
            "t": t,
            "detection": det.__dict__,
            "actions": [a.__dict__ for a in actions],
            "ground_truth_active_attacks": [
                inj.name for inj in injectors if inj.active(t)
            ],
            "step_latency_ms": step_latencies_ms[-1],
        }
        # Full decision provenance for steps on which the agent pipeline ran.
        # ``_last_decision["t"] == t`` guards against re-logging a stale state
        # from an earlier tick. ``llm_traces`` is popped so each real LLM call
        # is recorded exactly once, with its backend/adapter/prompt hashes,
        # raw output, and fallback reason (None in strict LLM runs).
        ld = getattr(detector, "_last_decision", None)
        if isinstance(ld, dict) and ld.get("t") == t:
            rec["decision_trace"] = {
                "decision_id": ld.get("decision_id"),
                "proposed_action": ld.get("proposed_action"),
                "proposed_action_target": ld.get("proposed_action_target"),
                "confidence": ld.get("confidence"),
                "caution": ld.get("caution"),
                "caution_reason": ld.get("caution_reason"),
                "coordinator_decision": ld.get("coordinator_decision"),
                "coordinator_reason": ld.get("coordinator_reason"),
                "final_action": ld.get("final_action"),
                "final_action_target": ld.get("final_action_target"),
                "candidate_estimates": ld.get("impact_estimates"),
                "chosen_for_hitl": ld.get("chosen_for_hitl"),
                "hypothesis_cached": ld.get("hypothesis_cached", False),
                # Runtime-assurance provenance (safety_projection mode): the
                # safe candidate set, why each action was removed, and whether
                # the model's own proposal was the one executed.
                "projection": ld.get("projection"),
                "llm_traces": ld.pop("llm_traces", None),
            }
        decisions.append(rec)

    pd.DataFrame(rows).to_csv(out / "timeseries.csv", index=False)
    (out / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d, default=str) for d in decisions))
    if hitl_log:
        (out / "hitl.jsonl").write_text("\n".join(json.dumps(d) for d in hitl_log))
    if step_latencies_ms:
        arr = np.array(step_latencies_ms)
        (out / "runtime.json").write_text(json.dumps({
            "n_steps": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.quantile(arr, 0.50)),
            "p95_ms": float(np.quantile(arr, 0.95)),
            "p99_ms": float(np.quantile(arr, 0.99)),
            "max_ms": float(arr.max()),
        }, indent=2))
    manifest = {
        "scenario": cfg["name"],
        "detector": detector.name,
        "seed": seed,
        "config_hash": _config_hash(cfg),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--detector", required=True,
                    help="dotted path, e.g. baselines.rule_ids.adapter:RuleIDS")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mod_name, cls_name = args.detector.split(":")
    import importlib
    cls = getattr(importlib.import_module(mod_name), cls_name)
    detector = cls()
    out = run_scenario(args.config, detector, args.seed)
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
