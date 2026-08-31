"""End-to-end test: caution-flagged action is routed through the HITL gate
and either resolves with operator approval or falls back at the SLO."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .feeder import StubFeeder
from .harness import run_scenario
from .hitl import OperatorApprover, OperatorUnavailable
from .types import Action, Detection, EventLog, TelemetrySample


@dataclass
class FakeCautioningDetector:
    """Detector that always flags a `freeze_setpoint` proposal but never executes
    it (returns []), so the harness must route via the HITL gate."""
    name: str = "fake_cautioning"
    _last_decision: dict = field(default_factory=dict)

    def step(self, t, telemetry, events):
        # Only flag once at t=10s so we have a single approval to track
        if abs(t - 10.0) < 1e-6:
            self._last_decision = {
                "decision_id": "test-decision-1",
                "caution": True,
                "proposed_action": "freeze_setpoint",
                "proposed_action_target": "INV_1",
                "affected_asset": "INV_1",
                "confidence": 0.65,
                "rationale": "test",
            }
            return Detection(asset="INV_1", attack_class="command_spoof",
                             confidence=0.65, rationale="test"), []
        self._last_decision = {}
        return Detection(asset=None, attack_class="none", confidence=0.0), []


def _scenario_cfg(tmp_path: Path) -> Path:
    cfg = {
        "name": "hitl_smoke",
        "feeder": "feeder.dss",
        "dt_s": 1.0,
        "duration_s": 80,                        # > slo_s of 60
        "monitored_buses": ["BUS1"],
        "ders": [{"id": "INV_1", "type": "pv", "max_kw": 100.0}],
        "attacks": [],
        "ground_truth": {"attack_class": "none"},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_approver_resolves_within_slo(tmp_path):
    cfg = _scenario_cfg(tmp_path)
    feeder = StubFeeder(monitored_buses=["BUS1"], ders=[{"id": "INV_1", "max_kw": 100.0}])
    op = OperatorApprover(response_s=20, slo_s=60)
    out = run_scenario(str(cfg), FakeCautioningDetector(), seed=0,
                       out_root=str(tmp_path / "runs"), feeder=feeder, operator=op)
    log = [json.loads(l) for l in (out / "hitl.jsonl").read_text().splitlines() if l]
    assert len(log) == 1
    assert log[0]["verdict"] == "approved"
    assert log[0]["actions"][0]["name"] == "freeze_setpoint"
    assert log[0]["t"] == 30.0  # submitted t=10, approved at t=10+20


def test_unavailable_falls_back_at_slo(tmp_path):
    cfg = _scenario_cfg(tmp_path)
    feeder = StubFeeder(monitored_buses=["BUS1"], ders=[{"id": "INV_1", "max_kw": 100.0}])
    op = OperatorUnavailable(slo_s=60)
    out = run_scenario(str(cfg), FakeCautioningDetector(), seed=0,
                       out_root=str(tmp_path / "runs"), feeder=feeder, operator=op)
    log = [json.loads(l) for l in (out / "hitl.jsonl").read_text().splitlines() if l]
    assert len(log) == 1
    assert log[0]["verdict"] == "slo_expired"
    assert log[0]["actions"][0]["name"] == "freeze_setpoint"
    assert log[0]["t"] == 70.0  # submitted t=10, slo expired at t=10+60
