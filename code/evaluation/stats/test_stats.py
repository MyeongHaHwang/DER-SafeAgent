"""Tests for the statistics module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .bootstrap import paired_bootstrap_ci
from .multitest import holm_correct
from .aggregate import aggregate_seeds


def test_holm_basic():
    out = holm_correct([0.001, 0.04, 0.5], alpha=0.05)
    assert out[0]["reject"] is True
    assert out[2]["reject"] is False


def test_aggregate_runs_per_group():
    df = pd.DataFrame({
        "scenario": ["a"] * 6,
        "detector": ["x"] * 3 + ["y"] * 3,
        "seed": [0, 1, 2, 0, 1, 2],
        "ens_kwh": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
    })
    out = aggregate_seeds(df, "ens_kwh")
    assert set(out["detector"]) == {"x", "y"}
    assert out.loc[out["detector"] == "x", "mean"].item() == pytest.approx(2.0)


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(20):
        for det, mean in [("a", 1.0), ("b", 2.0)]:
            rows.append({"scenario": "S", "seed": s, "detector": det,
                         "ens_kwh": float(rng.normal(mean, 0.1))})
    df = pd.DataFrame(rows)
    res = paired_bootstrap_ci(df, "ens_kwh", "a", "b", n_boot=2000, seed=0)
    assert res["point_estimate"] < 0  # method "a" delivers less ENS
    assert res["ci_high"] < 0
    assert res["p_one_sided"] < 0.05
