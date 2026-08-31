"""Unit tests for the external-benchmark label mapping."""
from __future__ import annotations

import pytest

from .label_mapping import DER_CLASSES, REGISTRY, coverage, to_der_class


@pytest.mark.parametrize("dataset", list(REGISTRY.keys()))
def test_every_label_maps_into_taxonomy(dataset):
    for src, (der, cap) in REGISTRY[dataset].items():
        assert der in DER_CLASSES, f"{dataset}/{src} → {der} not in taxonomy"
        assert 0.0 <= cap <= 1.0


def test_unknown_dataset_returns_none():
    der, cap = to_der_class("does_not_exist", "DoS")
    assert der == "none" and cap == 0.0


def test_unknown_label_falls_back_to_none():
    der, cap = to_der_class("unsw_mg24", "AlienLabel")
    assert der == "none"


def test_case_insensitive_unsw():
    der1, _ = to_der_class("unsw_mg24", "DoS")
    der2, _ = to_der_class("unsw_mg24", "dos")
    assert der1 == "dos" and der2 == "dos"


def test_coverage_includes_every_class_key():
    cov = coverage("unsw_mg24")
    assert set(cov.keys()) == set(DER_CLASSES)
