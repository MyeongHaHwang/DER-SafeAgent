"""Map source-dataset attack labels to the DER-SecAgent six-class taxonomy.

Sources covered:
    UNSW-MG24 (NSL-KDD-style) --- the only dataset with a shipped adapter.
    EPIC / SWaT / WADI / TON_IoT --- mappings provided here for when the
    user supplies a CSV; the run-time adapter still reports "not run" if the
    raw files are absent (see ``dataset_adapter.py``).

The DER-SecAgent taxonomy is::
    {none, fdi, replay, command_spoof, dos, firmware}
"""
from __future__ import annotations

from typing import Iterable

DER_CLASSES: tuple[str, ...] = (
    "none", "fdi", "replay", "command_spoof", "dos", "firmware",
)
# Extended taxonomy used for selective-prediction / abstention metrics.
DER_CLASSES_EXTENDED: tuple[str, ...] = DER_CLASSES + ("unknown_or_unmapped",)

# UNSW-MG24 / NSL-KDD-style → DER class. Confidence cap reflects how
# precisely the source label maps onto a cyber-physical attack class.
UNSW_MG24: dict[str, tuple[str, float]] = {
    "DoS":        ("dos",            1.00),
    "Probe":      ("none",           0.50),
    "R2L":        ("command_spoof",  0.55),
    "U2R":        ("firmware",       0.55),
    "MITM":       ("replay",         0.70),
    "BruteForce": ("command_spoof",  0.55),
}

# SWaT (Goh et al., 2017): attacks are labelled by "Attack #1" .. "Attack #36".
# We expose a coarse override here keyed by the label string itself.
SWAT: dict[str, tuple[str, float]] = {
    "Normal":              ("none",           1.00),
    "MITM":                ("replay",         0.70),
    "Sensor_Spoofing":     ("fdi",            0.85),
    "Setpoint_Tampering":  ("command_spoof",  0.85),
    "Denial_of_Service":   ("dos",            0.95),
    "Firmware_Tamper":     ("firmware",       0.85),
}

# WADI uses similar labels; reuse SWaT mapping by default and override by
# label-substring at adapter time.
WADI: dict[str, tuple[str, float]] = dict(SWAT)

# TON_IoT (Moustafa, 2020): wider label set; only the relevant slice maps to
# our taxonomy.
TON_IOT: dict[str, tuple[str, float]] = {
    "normal":         ("none",           1.00),
    "dos":            ("dos",            0.95),
    "ddos":           ("dos",            0.95),
    "injection":      ("command_spoof",  0.65),
    "mitm":           ("replay",         0.70),
    "ransomware":     ("firmware",       0.50),
    "scanning":       ("none",           0.40),
    "password":       ("command_spoof",  0.50),
    "xss":            ("command_spoof",  0.40),
    "backdoor":       ("firmware",       0.55),
}

# EPIC labels are scenario-dependent; treat as pass-through with low cap
# unless the user provides a custom map.
EPIC: dict[str, tuple[str, float]] = {
    "Normal":  ("none", 1.00),
    "Attack":  ("dos",  0.40),
}

REGISTRY: dict[str, dict[str, tuple[str, float]]] = {
    "unsw_mg24": UNSW_MG24,
    "swat":      SWAT,
    "wadi":      WADI,
    "ton_iot":   TON_IOT,
    "epic":      EPIC,
}


def to_der_class(dataset: str, source_label: str) -> tuple[str, float]:
    """Return ``(der_class, confidence_cap)`` for the supplied source label."""
    table = REGISTRY.get(dataset.lower())
    if table is None:
        return ("none", 0.0)
    if source_label in table:
        return table[source_label]
    # case-insensitive fallback
    s = source_label.lower()
    for k, v in table.items():
        if k.lower() == s:
            return v
    return ("none", 0.0)


def all_der_classes() -> tuple[str, ...]:
    return DER_CLASSES


def coverage(dataset: str) -> dict[str, list[str]]:
    """Reverse the mapping: which source labels collapse to each DER class."""
    table = REGISTRY.get(dataset.lower(), {})
    out: dict[str, list[str]] = {c: [] for c in DER_CLASSES}
    for src, (der, _cap) in table.items():
        out.setdefault(der, []).append(src)
    return out
