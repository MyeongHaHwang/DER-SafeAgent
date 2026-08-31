"""Shared types for the co-simulation harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TelemetrySample:
    t: float
    bus_voltages_pu: dict[str, float]
    freq_hz: float
    der_p_kw: dict[str, float]
    der_p_avail_kw: dict[str, float]
    load_demand_kw: float
    load_served_kw: float


@dataclass
class EventLog:
    t: float
    source: str          # "ied" | "dnp3" | "iec61850" | "host" | ...
    kind: str            # "telemetry" | "command" | "alarm" | "auth" | ...
    payload: dict[str, Any]
    tampered: bool = False  # ground truth — never visible to a runtime-safe detector
    # Observable protocol sequence number (v3): a monotonically increasing
    # per-source counter (IEC 61850 sqNum / DNP3 sequence analogue), carried
    # on the wire and therefore visible to the detector. A replay re-emits
    # captured frames with their *old* seq, producing an observable
    # regression. Default None keeps legacy events and the duplicate-payload
    # key unchanged, so all published (v2) results reproduce byte-for-byte.
    seq: int | None = None


@dataclass
class Detection:
    asset: str | None
    attack_class: str    # one of {"none", "fdi", "replay", "command_spoof", "dos", "firmware"}
    confidence: float
    rationale: str = ""


@dataclass
class Action:
    """A mitigation primitive issued by the detector / agent."""
    name: str            # see mitigations.py
    target: str          # asset id
    params: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    """Common interface for proposed agent + all baselines."""

    name: str

    def step(
        self,
        t: float,
        telemetry: TelemetrySample,
        events: list[EventLog],
    ) -> tuple[Detection, list[Action]]: ...
