"""Deadline-aware deployment evaluation (revision P0).

Two evaluation modes are kept explicitly separate:

* **Decision-quality mode** (``run_llm_in_loop.py``): the model-derived decision
  is applied once inference returns, and the wall-clock latency is recorded.
  This measures decision correctness and safety containment.

* **Deadline-aware deployment mode** (this module): the deterministic fast path
  acts immediately at detection time; the LLM runs concurrently; the
  LLM-informed action replaces the fast-path action *only* if it arrives before
  the SLO **and** passes every safety check. Latency is not hidden from the
  physical outcome — the fast-path action is what is in force for the whole
  interval [t_detect, t_detect + latency], and if the SLO expires first the
  fast-path action stands for the rest of the incident.

The physical consequence is recomputed by re-running the harness with a
*deadline wrapper* detector that replays the already-recorded LLM decision at
its measured arrival time. This reuses the real measured latency rather than
assuming one, and requires no additional GPU time.

Run:
  python3 -m code.evaluation.llm_in_loop.deadline_aware \
      --systems der_secagent_qwen_lora_k1,der_secagent_llama_lora_k1 \
      --slos 1,5,15,30,60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..physical_curves import sweep
from ...baselines.deterministic_energy_policy.adapter import DeterministicEnergyPolicy
from ...simulation.feeder import StubFeeder
from ...simulation.harness import run_scenario
from ...simulation.types import Action, Detection, EventLog, TelemetrySample

REVISION_TAG = "ijcip_revision_r1r2_20260805"
RES = Path("code/results") / REVISION_TAG / "llm_in_loop"
OUT = Path("code/results") / REVISION_TAG / "deadline_aware"
MANIFEST = Path("code/configs") / REVISION_TAG / "scenario_manifest.csv"


class DeadlineAwareDetector:
    """Deterministic fast path + replayed LLM decision under an SLO.

    ``llm_events`` is a list of ``(t_request, latency_s, final_action, target,
    passed_safety)`` tuples recovered from a completed decision-quality run.
    Before the LLM result arrives, the deterministic policy owns the action
    surface. When it arrives, it is adopted only if
    ``latency_s <= slo_s`` and it passed the safety pipeline.
    """

    def __init__(self, llm_events: list[dict], slo_s: float, name: str):
        self.name = name
        self.slo_s = slo_s
        self._fast = DeterministicEnergyPolicy()
        self._events = sorted(llm_events, key=lambda e: e["t_request"])
        self._applied: set[int] = set()
        self._last_decision = None
        self.n_llm_adopted = 0
        self.n_llm_expired = 0
        self.n_llm_rejected_by_safety = 0
        self.n_llm_advised_no_action = 0

    def step(self, t: float, telemetry: TelemetrySample,
             events: list[EventLog]) -> tuple[Detection, list[Action]]:
        det, actions = self._fast.step(t, telemetry, events)
        for i, ev in enumerate(self._events):
            if i in self._applied:
                continue
            arrival = ev["t_request"] + ev["latency_s"]
            if t + 1e-9 < arrival:
                continue                      # still pending: fast path stands
            self._applied.add(i)
            if ev["latency_s"] > self.slo_s:
                self.n_llm_expired += 1       # SLO expired: fast path stands
                continue
            if not ev["passed_safety"]:
                # The Coordinator withheld the action (HITL escalation or
                # withhold): the safety layer stopped it reaching the feeder.
                self.n_llm_rejected_by_safety += 1
                continue
            if ev["final_action"] in (None, "no_op"):
                # The model-informed decision was "take no action". This is
                # distinct from being blocked: nothing was proposed to adopt,
                # so the fast-path action simply stands.
                self.n_llm_advised_no_action += 1
                continue
            self.n_llm_adopted += 1
            actions = list(actions) + [Action(name=ev["final_action"],
                                              target=ev["target"] or "")]
        return det, actions


def _recover_llm_events(run_dir: Path) -> list[dict]:
    """Read a completed decision-quality run and extract, per LLM consultation,
    the request time, the *measured* end-to-end inference latency, and the
    post-safety action."""
    out = []
    for line in (run_dir / "decisions.jsonl").read_text().splitlines():
        rec = json.loads(line)
        dt = rec.get("decision_trace")
        if not dt or not dt.get("llm_traces"):
            continue
        latency_s = sum(t["latency_ms"] for t in dt["llm_traces"]) / 1000.0
        out.append({
            "t_request": float(rec["t"]),
            "latency_s": latency_s,
            "final_action": dt.get("final_action"),
            "target": dt.get("final_action_target"),
            # An action reaches execution only if the Coordinator auto-approved
            # it; hitl_required / withhold means the safety layer stopped it.
            "passed_safety": dt.get("coordinator_decision") == "auto",
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True)
    ap.add_argument("--slos", default="1,5,15,30,60")
    args = ap.parse_args()
    slos = [float(s) for s in args.slos.split(",")]
    man = pd.read_csv(MANIFEST).set_index("scenario_id")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for system in args.systems.split(","):
        runs_root = RES / system / "runs"
        if not runs_root.exists():
            print(f"[skip] no decision-quality runs for {system}")
            continue
        for man_path in sorted(runs_root.rglob("manifest.json")):
            run_dir = man_path.parent
            meta = json.loads(man_path.read_text())
            scen = meta["scenario"]
            if scen not in man.index:
                continue
            events = _recover_llm_events(run_dir)
            cfg_path = man.loc[scen, "config_path"]
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            for slo in slos:
                det = DeadlineAwareDetector(events, slo, f"{system}_slo{int(slo)}")
                feeder = StubFeeder(monitored_buses=cfg["monitored_buses"],
                                    ders=cfg["ders"],
                                    base_load_kw=float(cfg.get("base_load_kw", 1000.0)))
                out_root = OUT / "runs" / system / f"slo{int(slo)}"
                rd = out_root / scen / det.name / "seed0"
                if not (rd / "manifest.json").exists():
                    run_scenario(cfg_path, det, 0, out_root=str(out_root), feeder=feeder,
                                 extra_manifest={
                                     "revision_tag": REVISION_TAG,
                                     "evaluation_mode": "deadline_aware",
                                     "slo_s": slo,
                                     "source_decision_quality_run": str(run_dir),
                                     "llm_backend": meta.get("llm_backend"),
                                     "llm_adapter_sha": meta.get("llm_adapter_sha"),
                                     "llm_substrate": meta.get("llm_substrate"),
                                 })
                phys = sweep(rd, np.linspace(0.0, 1.0, 11))
                at05 = phys[phys["threshold"].between(0.49, 0.51)].iloc[0]
                lat = [e["latency_s"] for e in events]
                # Counts are derived from the events and the SLO rather than
                # from the detector's counters, so they are correct whether the
                # run executed now or was resumed from a completed run.
                n_expired = sum(1 for e in events if e["latency_s"] > slo)
                in_time = [e for e in events if e["latency_s"] <= slo]
                n_blocked = sum(1 for e in in_time if not e["passed_safety"])
                n_noaction = sum(1 for e in in_time if e["passed_safety"]
                                 and e["final_action"] in (None, "no_op"))
                n_adopted = len(in_time) - n_blocked - n_noaction
                rows.append({
                    "system": system, "scenario": scen, "slo_s": slo,
                    "n_llm_consultations": len(events),
                    "n_adopted": n_adopted,
                    "n_slo_expired": n_expired,
                    "n_rejected_by_safety": n_blocked,
                    "n_advised_no_action": n_noaction,
                    "frac_within_slo": (float(np.mean([l <= slo for l in lat]))
                                        if lat else float("nan")),
                    "mean_llm_latency_s": float(np.mean(lat)) if lat else float("nan"),
                    "max_llm_latency_s": float(np.max(lat)) if lat else float("nan"),
                    "ens_kwh": float(at05["ens_kwh"]),
                    "curt_kwh": float(at05["curt_kwh"]),
                    "voltage_frac": float(at05["voltage_frac"]),
                    "ramp_violations": float(at05["ramp_violations"]),
                    "llm_substrate": meta.get("llm_substrate"),
                })
            print(f"[done] {system} {scen} ({len(events)} consultations)")

    df = pd.DataFrame(rows)
    p = OUT / "slo_outcomes.csv"
    if p.exists():
        df = (pd.concat([pd.read_csv(p), df], ignore_index=True)
              .drop_duplicates(subset=["system", "scenario", "slo_s"], keep="last"))
    df.to_csv(p, index=False)
    print(f"deadline-aware outcomes -> {p} ({len(df)} rows)")


if __name__ == "__main__":
    main()
