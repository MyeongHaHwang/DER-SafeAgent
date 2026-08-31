"""Smoke test for the harness using StubFeeder (no OpenDSS required)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from .feeder import StubFeeder
from .harness import run_scenario
from .types import Action, Detection, EventLog, TelemetrySample


class NoOpDetector:
    name = "noop"

    def step(self, t: float, telemetry: TelemetrySample, events: list[EventLog]):
        return Detection(asset=None, attack_class="none", confidence=0.0), []


def test_harness_runs_end_to_end(tmp_path):
    cfg = {
        "name": "smoke",
        "feeder": "feeder.dss",
        "dt_s": 1.0,
        "duration_s": 5,
        "monitored_buses": ["BUS1"],
        "ders": [{"id": "INV_1", "type": "pv", "max_kw": 100.0}],
        "attacks": [],
        "ground_truth": {"attack_class": "none"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"])
    out = run_scenario(str(cfg_path), NoOpDetector(), seed=0,
                       out_root=str(tmp_path / "runs"), feeder=feeder)
    assert (out / "timeseries.csv").exists()
    assert (out / "decisions.jsonl").exists()
    assert (out / "manifest.json").exists()
