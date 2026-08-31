"""Bidirectional mapping between 2025 NSL-KDD-style classes and 2026 DER classes.

Used when comparing on UNSW-MG24 (which inherits NSL-KDD-style labels) so the
2026 detector's outputs can be scored against the 2025 ground truth.

The mapping is many-to-many; we encode it conservatively. When a precise
mapping is impossible we map to the *most plausible* 2026 class with a
confidence cap of 0.5 so the comparison is not flattering.
"""
from __future__ import annotations

from .states import AttackClass


# 2025 (UNSW-MG24 / NSL-KDD) → 2026 DER cyber-physical
KDD_TO_DER: dict[str, tuple[AttackClass, float]] = {
    "DoS":        (AttackClass.DOS, 1.0),
    "Probe":      (AttackClass.NONE, 0.5),         # Probe is reconnaissance, no direct DER class
    "R2L":        (AttackClass.COMMAND_SPOOF, 0.5),
    "U2R":        (AttackClass.FIRMWARE, 0.5),     # privilege escalation can lead to firmware tamper
    "MITM":       (AttackClass.REPLAY, 0.7),
    "BruteForce": (AttackClass.COMMAND_SPOOF, 0.5),
}


# 2026 → 2025 (best NSL-KDD-style label, lossy)
DER_TO_KDD: dict[AttackClass, str] = {
    AttackClass.NONE:          "Probe",
    AttackClass.FDI:           "MITM",
    AttackClass.REPLAY:        "MITM",
    AttackClass.COMMAND_SPOOF: "R2L",
    AttackClass.DOS:           "DoS",
    AttackClass.FIRMWARE:      "U2R",
}


def map_kdd_to_der(label: str) -> tuple[AttackClass, float]:
    """Return (der_class, confidence_cap)."""
    return KDD_TO_DER.get(label, (AttackClass.NONE, 0.0))


def map_der_to_kdd(label: AttackClass) -> str:
    return DER_TO_KDD.get(label, "Probe")
