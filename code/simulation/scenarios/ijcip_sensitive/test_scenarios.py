"""Sanity tests for the sensitive-scenario library."""
from __future__ import annotations

from code.simulation.scenarios.ijcip_sensitive import SCENARIOS, SensitiveScenario


def test_eight_scenarios_present():
    assert len(SCENARIOS) >= 8
    names = {s.name for s in SCENARIOS}
    expected = {
        "ambiguous_fdi_vs_command_spoof",
        "low_confidence_high_impact_action",
        "missing_asset_high_impact",
        "rationale_action_disagreement",
        "replay_stale_telemetry",
        "dos_persistent_freeze",
        "injected_wrong_llm_recommendation",
        "objective_tradeoff_case",
    }
    assert expected.issubset(names)


def test_every_scenario_has_required_fields():
    legal_keys = {"safe_actions", "must_not_action", "should_abstain",
                   "should_escalate_hitl", "schema_must_validate", "max_tier"}
    for s in SCENARIOS:
        assert s.name and s.description
        assert s.telemetry is not None
        assert isinstance(s.events, list)
        for k in s.expected_behavior:
            assert k in legal_keys, f"unknown key {k} on {s.name}"
        assert s.ground_truth_class in {"none", "fdi", "replay",
                                            "command_spoof", "dos", "firmware",
                                            "unknown_or_unmapped"}


def test_must_not_action_uses_action_registry():
    legal = {"no_op", "freeze_setpoint", "throttle_ramp",
             "request_ied_revalidation", "isolate_inverter"}
    for s in SCENARIOS:
        for a in s.expected_behavior.get("must_not_action", []):
            assert a in legal
        for a in s.expected_behavior.get("safe_actions", []):
            assert a in legal
