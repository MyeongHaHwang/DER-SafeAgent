"""Runtime-assurance safety projection (revision `ijcip_final_revision`, E1).

Motivation
----------
The deployed Coordinator applies a class-override table: once an attack class
is decided, the executed action is fully determined, and the Hypothesis Agent's
``proposed_action`` is discarded. Under that policy the model has almost no
causal path to the physical outcome, which is why the previous revision
measured the full stack, a scripted stand-in and a ground-truth oracle as
producing *identical* ENS on every configuration. That is a property of the
policy, not a safety result about language models.

This module implements the alternative that makes the question answerable:

``legacy_class_override``
    Preserved unchanged for reference. Class decides action.

``safety_projection``
    The safety layer *removes* candidate actions that are infeasible,
    forbidden, irreversible-without-justification, or inappropriate for the
    predicted class. Whatever remains is the **safe candidate set**, and the
    model's own proposal is honoured if it lies inside that set. The
    deterministic policy acts only when the model cannot be used: invalid
    output, low confidence, timeout, or an empty safe set.

This is the standard runtime-assurance / safety-shield construction: the
verified component constrains the action space; the unverified component
chooses within it. The model gains a real causal path while every executed
action remains inside the bounded set.

Every decision records why each action was removed, so a reviewer can audit the
projection rather than take it on trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REGISTRY = ("no_op", "freeze_setpoint", "throttle_ramp",
            "request_ied_revalidation", "isolate_inverter")

IRREVERSIBLE = {"isolate_inverter"}

# Class-inappropriate actions: a telemetry-layer compromise must not be
# answered with a physical-layer disconnection, because the asset itself is
# healthy. These are *removals*, not substitutions.
CLASS_INAPPROPRIATE: dict[str, set[str]] = {
    "fdi": {"isolate_inverter"},
    "replay": {"isolate_inverter"},
    "none": {"isolate_inverter", "freeze_setpoint", "throttle_ramp",
             "request_ied_revalidation"},
}

# Actions that cannot mitigate a given class at all (they leave the mechanism
# untouched), so selecting one would be a no-op with a cost.
# Revision (P0-A/P1-B, tag ijcip_final_safeagent_20260810):
# request_ied_revalidation added for dos — the counterfactual rollouts show
# it is physically equivalent to no_op against a forced-zero outage (median
# regret ~90 kWh-eq on dev, confirmed on the frozen holdout), so honouring
# it against evidenced DoS stands the outage down in practice.
CLASS_INEFFECTIVE: dict[str, set[str]] = {
    "command_spoof": {"request_ied_revalidation"},
    "fdi": {"throttle_ramp"},
    "dos": {"throttle_ramp", "request_ied_revalidation"},
}

CONFIDENCE_FLOOR = 0.40          # below this the model is not trusted to choose
IRREVERSIBLE_CONFIDENCE = 0.70   # irreversible actions need more than that


@dataclass
class ProjectionResult:
    """Outcome of one projection, with full provenance."""
    executed_action: str
    safe_candidate_set: list[str]
    removed: dict[str, str] = field(default_factory=dict)   # action -> reason
    source: str = "model"          # model | deterministic_fallback | caution_gate
    fallback_reason: str | None = None
    hitl_required: bool = False
    model_proposal_honoured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_action": self.executed_action,
            "safe_candidate_set": list(self.safe_candidate_set),
            "safety_veto_reason": dict(self.removed),
            "action_source": self.source,
            "fallback_reason": self.fallback_reason,
            "hitl_required": self.hitl_required,
            "model_proposal_honoured": self.model_proposal_honoured,
        }


def safe_candidates(attack_class: str, confidence: float,
                    severity: float) -> tuple[list[str], dict[str, str]]:
    """Return (allowed actions, {removed action: reason}).

    ``no_op`` is always retained: doing nothing is always feasible, and keeping
    it in the set is what allows the projection to express "no action is
    warranted" rather than forcing a mitigation.
    """
    removed: dict[str, str] = {}
    allowed: list[str] = []
    for a in REGISTRY:
        if a != "no_op" and severity < 0.30:
            removed[a] = "severity below action threshold"
            continue
        if a in CLASS_INAPPROPRIATE.get(attack_class, set()):
            removed[a] = f"class-inappropriate for {attack_class}"
            continue
        if a in CLASS_INEFFECTIVE.get(attack_class, set()):
            removed[a] = f"ineffective against {attack_class}"
            continue
        if a in IRREVERSIBLE and confidence < IRREVERSIBLE_CONFIDENCE:
            removed[a] = (f"irreversible action below confidence floor "
                          f"({confidence:.2f} < {IRREVERSIBLE_CONFIDENCE})")
            continue
        allowed.append(a)
    if "no_op" not in allowed:
        allowed.insert(0, "no_op")
        removed.pop("no_op", None)
    return allowed, removed


def deterministic_choice(attack_class: str, allowed: list[str]) -> str:
    """The fallback the shield uses when the model cannot be trusted.

    This is the legacy class-preference order, intersected with the safe set,
    so the fallback is never less safe than the projection allows.
    """
    preference = {
        "command_spoof": ["freeze_setpoint", "throttle_ramp", "no_op"],
        "fdi": ["freeze_setpoint", "request_ied_revalidation", "no_op"],
        "replay": ["request_ied_revalidation", "freeze_setpoint", "no_op"],
        "dos": ["request_ied_revalidation", "freeze_setpoint", "no_op"],
        "firmware": ["request_ied_revalidation", "freeze_setpoint", "no_op"],
        "none": ["no_op"],
    }.get(attack_class, ["no_op"])
    for a in preference:
        if a in allowed:
            return a
    return "no_op"


def project(attack_class: str, proposed_action: str | None, confidence: float,
            severity: float, cautioned: bool = False,
            parse_ok: bool = True,
            det_class_hint: str | None = None) -> ProjectionResult:
    """Apply the safety shield and let the model choose inside it.

    ``det_class_hint`` — the class implied by protocol-level deterministic
    evidence (tampered flags, forged commands, duplicate/stale telemetry,
    persistent freeze), or None when no such evidence exists. Revision note
    (P1-B, tag ijcip_final_safeagent_20260810): a DoS freezes the whole
    telemetry sample at capture time, so the model is shown a copy of
    *normal* data and proposes ``no_op`` with high plausibility; honouring
    that proposal stands down a deterministically evidenced incident and
    the outage runs unmitigated. The symmetric counterpart of the
    irreversible-action rule therefore applies: just as model output cannot
    *authorise* an irreversible action, it cannot *veto* an incident that
    deterministic evidence has established. A no-incident/no-action
    proposal against protocol evidence is answered by the deterministic
    fallback, computed under the evidence-implied class.
    """
    if (det_class_hint and det_class_hint != "none" and parse_ok
            and (proposed_action in (None, "no_op") or attack_class == "none")):
        # Safe set under the evidence-implied class; confidence=0 keeps any
        # irreversible primitive out of the fallback's reach.
        allowed_d, removed_d = safe_candidates(det_class_hint, 0.0,
                                               max(severity, 0.31))
        act = deterministic_choice(det_class_hint, allowed_d)
        return ProjectionResult(
            act, allowed_d, removed_d, "deterministic_fallback",
            "model proposed stand-down against deterministic protocol "
            "evidence; the model may not veto an evidenced incident")

    allowed, removed = safe_candidates(attack_class, confidence, severity)

    if not parse_ok or proposed_action is None:
        act = deterministic_choice(attack_class, allowed)
        return ProjectionResult(act, allowed, removed, "deterministic_fallback",
                                "model output unusable (parse failure)")
    if proposed_action not in REGISTRY:
        act = deterministic_choice(attack_class, allowed)
        return ProjectionResult(act, allowed, removed, "deterministic_fallback",
                                f"proposed action {proposed_action!r} outside registry")
    if confidence < CONFIDENCE_FLOOR:
        act = deterministic_choice(attack_class, allowed)
        return ProjectionResult(act, allowed, removed, "deterministic_fallback",
                                f"confidence {confidence:.2f} below floor "
                                f"{CONFIDENCE_FLOOR}")
    if proposed_action not in allowed:
        # The shield vetoed the model's choice. Fall back to the safest
        # available action and record why the proposal was rejected.
        act = deterministic_choice(attack_class, allowed)
        return ProjectionResult(
            act, allowed, removed, "deterministic_fallback",
            f"proposal {proposed_action!r} removed: "
            f"{removed.get(proposed_action, 'not in safe set')}")

    # The model's proposal survives the shield.
    #
    # Revision note (E6): an earlier version auto-executed an irreversible
    # primitive whenever the predicted class did not explicitly forbid it and
    # the stated confidence cleared the floor. Adversarial evaluation showed
    # that is exploitable: injection prompts drive a high-confidence
    # command_spoof/dos prediction proposing isolate_inverter, and it executed
    # on 52 of 70 cases (containment 0.00 across five perturbation families).
    # Model-stated confidence is attacker-influenced and cannot gate an
    # irreversible physical action. An irreversible primitive proposed by the
    # model is therefore ALWAYS escalated, never auto-executed.
    if proposed_action in IRREVERSIBLE:
        return ProjectionResult("no_op", allowed, removed, "caution_gate",
                                "irreversible action proposed by the model is "
                                "never auto-executed; escalated for human review",
                                hitl_required=True)
    return ProjectionResult(proposed_action, allowed, removed, "model",
                            None, False, model_proposal_honoured=True)
