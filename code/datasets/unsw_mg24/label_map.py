"""UNSW-MG24 attack label → DER cyber-physical class mapping.

UNSW-MG24 inherits NSL-KDD-style labels (DoS, Probe, R2L, U2R, MITM,
BruteForce). Our 2026 taxonomy is cyber-physical-grounded
(none/fdi/replay/command_spoof/dos/firmware). This file is the
authoritative mapping; it is also exposed in
`code/Multi_AI_Agent/attack_class_map.py` for the agent stack but defined
here so the dataset loader is self-contained.
"""
from __future__ import annotations

KDD_TO_DER: dict[str, tuple[str, float]] = {
    "DoS":        ("dos", 1.0),
    "Probe":      ("none", 0.5),          # reconnaissance — no direct DER class
    "R2L":        ("command_spoof", 0.5),
    "U2R":        ("firmware", 0.5),
    "MITM":       ("replay", 0.7),
    "BruteForce": ("command_spoof", 0.5),
}


def to_der_class(kdd_label: str) -> tuple[str, float]:
    """Returns (der_class, confidence_cap)."""
    return KDD_TO_DER.get(kdd_label, ("none", 0.0))
