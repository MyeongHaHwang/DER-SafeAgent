"""Tests for the UNSW-MG24 loader and feeder shim using the synthetic mock."""
from __future__ import annotations

from pathlib import Path

from .loader import UNSWMG24Feeder, UNSWMG24Loader, synthesize_alerts


def test_synthesize_and_load(tmp_path):
    csv_path = tmp_path / "alerts_mock.csv"
    synthesize_alerts(str(csv_path), n_alerts=20, seed=0, duration_s=300)
    loader = UNSWMG24Loader(str(csv_path))
    assert loader.total_alerts == 20
    assert 60 <= loader.time_horizon_s <= 300


def test_feeder_emits_alarm_events(tmp_path):
    csv_path = tmp_path / "alerts_mock.csv"
    synthesize_alerts(str(csv_path), n_alerts=15, seed=1, duration_s=200)
    loader = UNSWMG24Loader(str(csv_path))
    feeder = UNSWMG24Feeder(loader=loader)
    # find the first non-empty bucket
    found = 0
    for t in range(0, 200):
        sample, events = feeder.read(float(t))
        if events:
            for ev in events:
                assert ev.kind == "alarm"
                assert "kdd_label" in ev.payload
                assert "der_class_hint" in ev.payload
            found += len(events)
    assert found == 15


def test_kdd_label_maps_to_der_class(tmp_path):
    csv_path = tmp_path / "alerts_mock.csv"
    synthesize_alerts(str(csv_path), n_alerts=200, seed=2)
    loader = UNSWMG24Loader(str(csv_path))
    feeder = UNSWMG24Feeder(loader=loader)
    # Spot-check that mapping fired (at least one event for each known class).
    classes_seen = set()
    for t in range(0, loader.time_horizon_s + 1):
        _, events = feeder.read(float(t))
        for ev in events:
            classes_seen.add(ev.payload["der_class_hint"])
    # Should see at least 'dos' and one of {none, command_spoof, replay, firmware}
    assert "dos" in classes_seen
    assert classes_seen & {"none", "command_spoof", "replay", "firmware"}
