"""Mitigation primitives the detector may request.

The harness applies these against the OpenDSS model. Each primitive is
documented with its expected energy-side effect so the agent's policy can
reason about cost (curtailment) vs. benefit (avoided ENS).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MitigationSpec:
    name: str
    description: str
    expected_side_effect: str   # human-readable; logged to results


REGISTRY: dict[str, MitigationSpec] = {
    "isolate_inverter": MitigationSpec(
        name="isolate_inverter",
        description="Open the breaker upstream of the named inverter.",
        expected_side_effect="curtails inverter output → +curtailment, possibly +ENS if load relied on it",
    ),
    "throttle_ramp": MitigationSpec(
        name="throttle_ramp",
        description="Reduce permitted ramp rate of inverter to the IEEE 1547 minimum.",
        expected_side_effect="caps fast power swings → -ramp_violations, marginal -ENS impact",
    ),
    "freeze_setpoint": MitigationSpec(
        name="freeze_setpoint",
        description="Reject DERMS setpoint commands; hold last validated setpoint.",
        expected_side_effect="defends command-spoof → tracking error grows but no harmful command applied",
    ),
    "request_ied_revalidation": MitigationSpec(
        name="request_ied_revalidation",
        description="Force IED to re-attest its firmware / re-handshake.",
        expected_side_effect="brief telemetry gap; no power impact unless attestation fails",
    ),
    "no_op": MitigationSpec(
        name="no_op",
        description="No mitigation issued.",
        expected_side_effect="zero curtailment cost; full attack impact passes through if real",
    ),
}


def is_valid(action_name: str) -> bool:
    return action_name in REGISTRY
