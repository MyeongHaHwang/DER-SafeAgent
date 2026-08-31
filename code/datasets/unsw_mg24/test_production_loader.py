"""Tests for the production UNSW-MG24 loader and label mapping."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from .label_mapping import (DER_CLASSES_EXTENDED, to_der_class,
                              to_der_class_strict)
from .production_loader import UnswProductionLoader


def test_label_mapping_high_confidence():
    der, cap = to_der_class("DoS")
    assert der == "dos" and cap >= 0.9


def test_label_mapping_unmapped_collapses_to_unknown():
    der, cap = to_der_class("Probe")
    assert der == "unknown_or_unmapped" and cap == 0.0


def test_label_mapping_unknown_label_is_unknown():
    der, cap = to_der_class("AlienLabel")
    assert der == "unknown_or_unmapped" and cap == 0.0


def test_extended_taxonomy_includes_unknown():
    assert "unknown_or_unmapped" in DER_CLASSES_EXTENDED


def test_strict_mapping_preserves_six_class():
    der, _ = to_der_class_strict("Probe")
    assert der in {"none", "fdi", "replay", "command_spoof", "dos", "firmware"}


def test_loader_falls_back_to_mock_when_unset(monkeypatch):
    monkeypatch.delenv("UNSW_MG24_ROOT", raising=False)
    loader = UnswProductionLoader()
    # The shipped mock CSV is the deepest fallback.
    assert loader.available()
    samples = list(loader.iter_samples(max_samples=10))
    assert len(samples) > 0
    for s in samples:
        assert s.der_class in DER_CLASSES_EXTENDED
        assert s.result_source in {"production", "mock"}


def test_loader_unavailable_when_no_files(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSW_MG24_ROOT", raising=False)
    fake = tmp_path / "missing.csv"
    loader = UnswProductionLoader(csv_path=str(fake))
    # If a non-existent override is given AND no fallbacks exist, the
    # loader must report unavailable rather than fabricate data.
    if not loader.available():
        assert loader.unavailable_reason
    else:
        # The repository ships a mock; the loader resolves it as final
        # fallback even if csv_path is bogus.
        assert loader.result_source == "mock"


def test_loader_uses_explicit_csv(tmp_path):
    p = tmp_path / "tiny.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["timestamp", "label"])
        w.writeheader()
        w.writerow({"timestamp": "5", "label": "DoS"})
        w.writerow({"timestamp": "9", "label": "Probe"})
    loader = UnswProductionLoader(csv_path=str(p))
    samples = list(loader.iter_samples())
    assert [s.der_class for s in samples] == ["dos", "unknown_or_unmapped"]
    assert loader.result_source == "production"
