"""Smoke test for the Multi-AI-Agent adapter (no GPU required).

Uses the LoRA server's discriminative heuristic fallback when transformers /
peft / adapter checkpoint are unavailable (e.g., CI).
"""
from __future__ import annotations

from ..simulation.types import EventLog, TelemetrySample
from .adapter import DERSecAgentDetector


def _telemetry(t: float, p: float = 100.0) -> TelemetrySample:
    return TelemetrySample(
        t=t,
        bus_voltages_pu={"BUS1": 1.0},
        freq_hz=60.0,
        der_p_kw={"INV_1": p},
        der_p_avail_kw={"INV_1": 200.0},
        load_demand_kw=500.0,
        load_served_kw=500.0,
    )


def test_does_not_invoke_when_quiet():
    det = DERSecAgentDetector()
    for k in range(10):
        d, actions = det.step(float(k), _telemetry(float(k), p=100.0), [])
    assert d.attack_class == "none"
    assert actions == []


def test_invokes_on_alarm_returns_structured_decision():
    det = DERSecAgentDetector()
    for k in range(10):
        det.step(float(k), _telemetry(float(k), p=100.0), [])
    ev_alarm = EventLog(t=10.0, source="iec61850", kind="alarm", payload={"why": "test"})
    ev_cmd = EventLog(t=10.0, source="dnp3", kind="command",
                      payload={"asset": "INV_1", "from": "DERMSx"})
    d, actions = det.step(10.0, _telemetry(10.0, p=100.0), [ev_alarm, ev_cmd])
    assert d.attack_class in {"none", "fdi", "replay", "command_spoof", "dos", "firmware"}
    assert isinstance(d.confidence, float)
    valid_actions = {"isolate_inverter", "throttle_ramp", "freeze_setpoint",
                     "request_ied_revalidation"}
    assert all(a.name in valid_actions for a in actions)
