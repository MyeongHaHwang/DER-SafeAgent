"""Human-In-The-Loop approval gate.

When the Caution Agent flags a proposed action, the harness routes it through
this gate. Three operator behaviours are supported:

  - `OperatorApprover`     — auto-approves after a configurable response delay.
  - `OperatorRejecter`     — auto-rejects after a delay (used in adversarial
                              ablations).
  - `OperatorUnavailable`  — never responds; the SLO triggers the fallback path.

Each gate maintains a queue of pending approvals keyed by `decision_id`. Every
tick `step(t)` advances pending entries; if `t - submitted_at >= response_s`
the operator's verdict resolves; if `t - submitted_at >= slo_s` and no verdict
yet, the SLO has expired and the harness applies the fallback policy.

The fallback policy is a `freeze_setpoint` aimed at the affected asset (a small
guaranteed-safe action that bounds harm) plus an on-call notification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from .types import Action


@dataclass
class PendingApproval:
    decision_id: str
    submitted_at: float
    proposed_action: Action
    affected_asset: str
    rationale: str


@dataclass
class ApprovalOutcome:
    decision_id: str
    verdict: Literal["approved", "rejected", "slo_expired"]
    actions: list[Action]
    note: str = ""


class HITLOperator(Protocol):
    name: str
    response_s: float
    slo_s: float

    def submit(self, t: float, pending: PendingApproval) -> None: ...
    def step(self, t: float) -> list[ApprovalOutcome]: ...


def _fallback_actions(p: PendingApproval) -> list[Action]:
    """Bounded-harm fallback used when the operator misses the SLO."""
    if not p.affected_asset:
        return []
    return [Action(
        name="freeze_setpoint",
        target=p.affected_asset,
        params={"reason": "HITL SLO expired; on-call paged",
                "original_action": p.proposed_action.name},
    )]


@dataclass
class OperatorApprover:
    name: str = "auto_approver"
    response_s: float = 30.0       # operator answers in 30 ticks (sec)
    slo_s: float = 60.0            # if no answer by 60 ticks, fallback
    _queue: list[PendingApproval] = field(default_factory=list)

    def submit(self, t: float, pending: PendingApproval) -> None:
        self._queue.append(pending)

    def step(self, t: float) -> list[ApprovalOutcome]:
        out: list[ApprovalOutcome] = []
        keep: list[PendingApproval] = []
        for p in self._queue:
            elapsed = t - p.submitted_at
            if elapsed >= self.response_s and elapsed < self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id,
                    verdict="approved",
                    actions=[p.proposed_action],
                    note=f"approved after {elapsed:.0f}s",
                ))
            elif elapsed >= self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id,
                    verdict="slo_expired",
                    actions=_fallback_actions(p),
                    note=f"SLO expired at {elapsed:.0f}s; fallback applied",
                ))
            else:
                keep.append(p)
        self._queue = keep
        return out


@dataclass
class OperatorRejecter:
    """Auto-rejects every escalation after `response_s`. Used to model an
    overly cautious operator posture in the IJCIP HITL sensitivity sweep."""
    name: str = "auto_rejecter"
    response_s: float = 30.0
    slo_s: float = 60.0
    _queue: list[PendingApproval] = field(default_factory=list)

    def submit(self, t: float, pending: PendingApproval) -> None:
        self._queue.append(pending)

    def step(self, t: float) -> list[ApprovalOutcome]:
        out: list[ApprovalOutcome] = []
        keep: list[PendingApproval] = []
        for p in self._queue:
            elapsed = t - p.submitted_at
            if elapsed >= self.response_s and elapsed < self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id, verdict="rejected",
                    actions=[], note=f"rejected after {elapsed:.0f}s"))
            elif elapsed >= self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id, verdict="slo_expired",
                    actions=_fallback_actions(p),
                    note=f"SLO expired at {elapsed:.0f}s; fallback applied"))
            else:
                keep.append(p)
        self._queue = keep
        return out


@dataclass
class OperatorModifier:
    """Operator approves the escalation but rewrites the proposed action
    to a safe fallback (``freeze_setpoint``)."""
    name: str = "auto_modifier"
    response_s: float = 30.0
    slo_s: float = 60.0
    _queue: list[PendingApproval] = field(default_factory=list)

    def submit(self, t: float, pending: PendingApproval) -> None:
        self._queue.append(pending)

    def step(self, t: float) -> list[ApprovalOutcome]:
        out: list[ApprovalOutcome] = []
        keep: list[PendingApproval] = []
        for p in self._queue:
            elapsed = t - p.submitted_at
            if elapsed >= self.response_s and elapsed < self.slo_s:
                modified = Action(
                    name="freeze_setpoint", target=p.affected_asset,
                    params={"original_action": p.proposed_action.name,
                              "modifier": "operator_safe_fallback"})
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id, verdict="approved",
                    actions=[modified],
                    note=f"approved-modified after {elapsed:.0f}s"))
            elif elapsed >= self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id, verdict="slo_expired",
                    actions=_fallback_actions(p),
                    note="SLO expired; fallback applied"))
            else:
                keep.append(p)
        self._queue = keep
        return out


@dataclass
class OperatorNoisy:
    """Operator that approves with probability ``approve_prob`` and rejects
    otherwise. Used to test sensitivity to flaky human review."""
    name: str = "auto_noisy"
    response_s: float = 30.0
    slo_s: float = 60.0
    approve_prob: float = 0.5
    seed: int = 0
    _queue: list[PendingApproval] = field(default_factory=list)

    def __post_init__(self) -> None:
        import random as _r
        self._rng = _r.Random(self.seed)

    def submit(self, t: float, pending: PendingApproval) -> None:
        self._queue.append(pending)

    def step(self, t: float) -> list[ApprovalOutcome]:
        out: list[ApprovalOutcome] = []
        keep: list[PendingApproval] = []
        for p in self._queue:
            elapsed = t - p.submitted_at
            if elapsed >= self.response_s and elapsed < self.slo_s:
                if self._rng.random() < self.approve_prob:
                    out.append(ApprovalOutcome(
                        decision_id=p.decision_id, verdict="approved",
                        actions=[p.proposed_action],
                        note=f"noisy-approved after {elapsed:.0f}s"))
                else:
                    out.append(ApprovalOutcome(
                        decision_id=p.decision_id, verdict="rejected",
                        actions=[],
                        note=f"noisy-rejected after {elapsed:.0f}s"))
            elif elapsed >= self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id, verdict="slo_expired",
                    actions=_fallback_actions(p),
                    note="SLO expired; fallback applied"))
            else:
                keep.append(p)
        self._queue = keep
        return out


@dataclass
class OperatorUnavailable:
    name: str = "unavailable_operator"
    response_s: float = float("inf")
    slo_s: float = 60.0
    _queue: list[PendingApproval] = field(default_factory=list)

    def submit(self, t: float, pending: PendingApproval) -> None:
        self._queue.append(pending)

    def step(self, t: float) -> list[ApprovalOutcome]:
        out: list[ApprovalOutcome] = []
        keep: list[PendingApproval] = []
        for p in self._queue:
            if t - p.submitted_at >= self.slo_s:
                out.append(ApprovalOutcome(
                    decision_id=p.decision_id,
                    verdict="slo_expired",
                    actions=_fallback_actions(p),
                    note="operator unavailable; SLO fallback applied",
                ))
            else:
                keep.append(p)
        self._queue = keep
        return out
