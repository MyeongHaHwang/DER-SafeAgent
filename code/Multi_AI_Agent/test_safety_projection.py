"""Tests for the runtime-assurance safety projection (E1)."""
from __future__ import annotations

import pytest

from .safety_projection import (CONFIDENCE_FLOOR, IRREVERSIBLE_CONFIDENCE,
                                REGISTRY, deterministic_choice, project,
                                safe_candidates)


def test_executed_action_is_always_in_the_registry():
    for cls in ("none", "fdi", "replay", "command_spoof", "dos", "firmware"):
        for proposed in list(REGISTRY) + ["rm -rf /", None, "shutdown"]:
            r = project(cls, proposed, 0.9, 0.9)
            assert r.executed_action in REGISTRY


def test_model_proposal_is_honoured_when_it_survives_the_shield():
    """The point of the projection: the model can actually choose."""
    r = project("command_spoof", "throttle_ramp", 0.9, 0.9)
    assert r.executed_action == "throttle_ramp"
    assert r.model_proposal_honoured and r.source == "model"


def test_model_can_select_different_actions_for_the_same_class():
    """Contrast with legacy_class_override, under which the class alone fixes
    the action. Here two different proposals give two different outcomes."""
    a = project("command_spoof", "freeze_setpoint", 0.9, 0.9).executed_action
    b = project("command_spoof", "throttle_ramp", 0.9, 0.9).executed_action
    assert a != b, "safety_projection must give the model a causal path"


def test_class_inappropriate_action_is_removed_not_substituted():
    r = project("fdi", "isolate_inverter", 0.95, 0.9)
    assert r.executed_action != "isolate_inverter"
    assert "isolate_inverter" in r.removed
    assert "class-inappropriate" in r.removed["isolate_inverter"]


def test_irreversible_proposal_is_never_auto_executed():
    """E6 regression: model-stated confidence is attacker-influenced, so it
    must not gate an irreversible physical action."""
    for cls in ("command_spoof", "dos", "firmware"):
        r = project(cls, "isolate_inverter", 0.99, 0.95)
        assert r.executed_action != "isolate_inverter"
        assert r.hitl_required, f"{cls}: irreversible proposal must escalate"


def test_irreversible_action_requires_high_confidence():
    low = project("command_spoof", "isolate_inverter", 0.5, 0.9)
    assert low.executed_action != "isolate_inverter"
    assert "confidence floor" in low.removed.get("isolate_inverter", "")
    # It may remain in the safe *candidate* set at high confidence, but E6
    # showed it must never be auto-executed from a model proposal.
    high = project("command_spoof", "isolate_inverter", 0.95, 0.9)
    assert "isolate_inverter" in high.safe_candidate_set
    assert high.executed_action != "isolate_inverter" and high.hitl_required


def test_low_confidence_routes_to_deterministic_fallback():
    r = project("command_spoof", "throttle_ramp", CONFIDENCE_FLOOR - 0.01, 0.9)
    assert r.source == "deterministic_fallback"
    assert "below floor" in (r.fallback_reason or "")


def test_unparseable_output_routes_to_deterministic_fallback():
    r = project("dos", None, 0.9, 0.9, parse_ok=False)
    assert r.source == "deterministic_fallback"
    assert r.executed_action in REGISTRY


def test_out_of_registry_proposal_never_executes():
    r = project("dos", "sudo reboot", 0.99, 0.9)
    assert r.executed_action in REGISTRY
    assert "outside registry" in (r.fallback_reason or "")


def test_benign_class_permits_only_no_op():
    allowed, removed = safe_candidates("none", 0.9, 0.9)
    assert allowed == ["no_op"]
    assert len(removed) == 4


def test_low_severity_permits_only_no_op():
    allowed, _ = safe_candidates("command_spoof", 0.9, 0.1)
    assert allowed == ["no_op"]


def test_no_op_is_always_available():
    for cls in ("none", "fdi", "replay", "command_spoof", "dos", "firmware"):
        for sev in (0.0, 0.5, 1.0):
            allowed, _ = safe_candidates(cls, 0.9, sev)
            assert "no_op" in allowed


def test_cautioned_irreversible_action_escalates_to_hitl():
    r = project("command_spoof", "isolate_inverter", 0.95, 0.9, cautioned=True)
    assert r.hitl_required and r.executed_action == "no_op"


def test_deterministic_fallback_never_leaves_the_safe_set():
    for cls in ("none", "fdi", "replay", "command_spoof", "dos", "firmware"):
        allowed, _ = safe_candidates(cls, 0.5, 0.9)
        assert deterministic_choice(cls, allowed) in allowed


def test_removal_reasons_are_recorded_for_audit():
    r = project("fdi", "throttle_ramp", 0.9, 0.9)
    assert r.removed, "every removal must carry a reason"
    assert all(isinstance(v, str) and v for v in r.removed.values())


def test_model_cannot_veto_evidenced_incident():
    """P1-B rule: a no_op/none proposal against deterministic protocol
    evidence falls back deterministically instead of standing down."""
    from .safety_projection import project
    r = project(attack_class="none", proposed_action="no_op", confidence=0.9,
                severity=0.7, det_class_hint="dos")
    assert r.source == "deterministic_fallback"
    assert r.executed_action != "no_op"
    assert "veto" in (r.fallback_reason or "")


def test_no_veto_rule_without_protocol_evidence():
    """Without a deterministic evidence hint, a no_op proposal is honoured."""
    from .safety_projection import project
    r = project(attack_class="none", proposed_action="no_op", confidence=0.9,
                severity=0.7, det_class_hint=None)
    assert r.executed_action == "no_op"


def test_veto_fallback_never_irreversible():
    from .safety_projection import project
    r = project(attack_class="none", proposed_action="no_op", confidence=0.99,
                severity=0.9, det_class_hint="command_spoof")
    assert r.executed_action != "isolate_inverter"
