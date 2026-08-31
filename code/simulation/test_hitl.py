"""Unit tests for the HITL approval gate."""
from __future__ import annotations

import pytest

from .hitl import (
    ApprovalOutcome,
    OperatorApprover,
    OperatorUnavailable,
    PendingApproval,
)
from .types import Action


def _pending(t: float = 0.0, action: str = "isolate_inverter",
             target: str = "INV_1") -> PendingApproval:
    return PendingApproval(
        decision_id="d1",
        submitted_at=t,
        proposed_action=Action(name=action, target=target),
        affected_asset=target,
        rationale="test",
    )


def test_approver_releases_after_response_delay():
    op = OperatorApprover(response_s=30, slo_s=60)
    op.submit(t=0.0, pending=_pending(t=0.0))
    # Before response delay: no outcome
    assert op.step(t=10.0) == []
    assert op.step(t=29.0) == []
    # At response delay: approved
    outs = op.step(t=30.0)
    assert len(outs) == 1
    assert outs[0].verdict == "approved"
    assert outs[0].actions[0].name == "isolate_inverter"


def test_approver_falls_back_at_slo():
    # Operator effectively unresponsive (response_s > slo_s)
    op = OperatorApprover(response_s=999, slo_s=60)
    op.submit(t=0.0, pending=_pending(t=0.0))
    assert op.step(t=30.0) == []
    outs = op.step(t=60.0)
    assert len(outs) == 1
    assert outs[0].verdict == "slo_expired"
    # Fallback action should be freeze_setpoint
    assert outs[0].actions[0].name == "freeze_setpoint"
    assert outs[0].actions[0].target == "INV_1"


def test_unavailable_only_emits_after_slo():
    op = OperatorUnavailable(slo_s=60)
    op.submit(t=0.0, pending=_pending(t=0.0))
    assert op.step(t=30.0) == []
    outs = op.step(t=60.0)
    assert outs[0].verdict == "slo_expired"
    assert outs[0].actions[0].name == "freeze_setpoint"


def test_queue_drains_after_resolution():
    op = OperatorApprover(response_s=10, slo_s=60)
    op.submit(t=0.0, pending=_pending(t=0.0))
    op.step(t=10.0)  # resolves
    # Queue should be empty now
    assert op.step(t=20.0) == []
