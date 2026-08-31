"""UNSW-MG24 → DER-SecAgent taxonomy mapping (production-grade).

Adds an explicit ``unknown_or_unmapped`` class so the detector can
abstain on labels whose physical meaning cannot be safely inferred.

The 2025 mapping in ``label_map.py`` collapses every UNSW-MG24 class
into the six-element DER taxonomy with a small confidence cap. That
behaviour is preserved in :func:`to_der_class_strict`; the new
:func:`to_der_class` returns ``unknown_or_unmapped`` for labels that
do not have a high-confidence cyber-physical equivalent.
"""
from __future__ import annotations

from typing import Iterable

DER_CLASSES_EXTENDED: tuple[str, ...] = (
    "none", "fdi", "replay", "command_spoof",
    "dos", "firmware", "unknown_or_unmapped",
)

# High-confidence mapping --- only labels that DER-SecAgent can safely
# act upon are mapped to a positive DER class. Lower-confidence
# packet-level labels collapse to ``unknown_or_unmapped`` so the
# detector abstains (or routes to HITL) instead of forcing an
# unsupported decision.
HIGH_CONFIDENCE_MAP: dict[str, tuple[str, float]] = {
    "DoS":              ("dos",            0.95),
    "DDoS":             ("dos",            0.95),
    "DDOS":             ("dos",            0.95),
    "Denial of Service":("dos",            0.95),
    "MITM":             ("replay",         0.70),
    "Man-in-the-Middle":("replay",         0.70),
    "Replay":           ("replay",         0.85),
    "Sensor_Spoofing":  ("fdi",            0.85),
    "FDI":              ("fdi",            0.90),
    "FDIA":             ("fdi",            0.90),
    "Setpoint_Tampering":("command_spoof", 0.85),
    "Command_Injection":("command_spoof",  0.80),
    "Firmware":         ("firmware",       0.80),
    "Firmware_Tamper":  ("firmware",       0.85),
    "Normal":           ("none",           1.00),
    "Benign":           ("none",           1.00),
}

# Low-confidence labels. We recognise them so they are *not* treated as
# unknown noise, but we deliberately route them to abstention.
LOW_CONFIDENCE_MAP: dict[str, str] = {
    "Probe":            "unknown_or_unmapped",
    "Reconnaissance":   "unknown_or_unmapped",
    "Scanning":         "unknown_or_unmapped",
    "R2L":              "unknown_or_unmapped",
    "U2R":              "unknown_or_unmapped",
    "BruteForce":       "unknown_or_unmapped",
    "Brute-Force":      "unknown_or_unmapped",
    "Password":         "unknown_or_unmapped",
    "XSS":              "unknown_or_unmapped",
    "SQL_Injection":    "unknown_or_unmapped",
    "Backdoor":         "unknown_or_unmapped",
    "Worm":             "unknown_or_unmapped",
    "Ransomware":       "unknown_or_unmapped",
    "Generic":          "unknown_or_unmapped",
    "Exploits":         "unknown_or_unmapped",
    "Analysis":         "unknown_or_unmapped",
    "Shellcode":        "unknown_or_unmapped",
    "Fuzzers":          "unknown_or_unmapped",
}


def to_der_class(label: str) -> tuple[str, float]:
    """High-confidence mapping with explicit abstention.

    Returns ``(der_class, confidence_cap)``. Unmapped labels collapse to
    ``unknown_or_unmapped`` with a cap of 0.0 so the detector cannot act
    on them with high confidence by accident.
    """
    if not label:
        return ("unknown_or_unmapped", 0.0)
    if label in HIGH_CONFIDENCE_MAP:
        return HIGH_CONFIDENCE_MAP[label]
    # case-insensitive HC fallback
    s = label.lower()
    for k, v in HIGH_CONFIDENCE_MAP.items():
        if k.lower() == s:
            return v
    if label in LOW_CONFIDENCE_MAP or s in {k.lower() for k in LOW_CONFIDENCE_MAP}:
        return ("unknown_or_unmapped", 0.0)
    return ("unknown_or_unmapped", 0.0)


def to_der_class_strict(label: str) -> tuple[str, float]:
    """Backward-compatible 6-class mapping (no ``unknown_or_unmapped``).

    Useful when computing macro-F1 on the original taxonomy.
    """
    der, cap = to_der_class(label)
    if der == "unknown_or_unmapped":
        # legacy collapse rule from `label_map.py`: probe → none, R2L →
        # command_spoof, U2R → firmware, brute-force → command_spoof.
        s = label.lower()
        if "probe" in s or "scan" in s or "reconnaissance" in s:
            return ("none", 0.50)
        if s == "r2l" or "brute" in s or "password" in s:
            return ("command_spoof", 0.50)
        if s == "u2r" or "ransom" in s or "shell" in s or "back" in s:
            return ("firmware", 0.50)
        return ("none", 0.0)
    return (der, cap)


def coverage(labels: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in DER_CLASSES_EXTENDED}
    for lbl in labels:
        der, _ = to_der_class(lbl)
        out.setdefault(der, []).append(lbl)
    return out
