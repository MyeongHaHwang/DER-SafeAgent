"""Tests for configuration-level and hierarchical inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchical import (configuration_level_paired, degenerate_mask,
                           hierarchical_bootstrap, holm)


def _frame(values: dict[tuple[str, str], list[float]]) -> pd.DataFrame:
    rows = []
    for (scen, det), vals in values.items():
        for seed, v in enumerate(vals):
            rows.append({"scenario": scen, "detector": det, "seed": seed,
                         "ens_kwh": v})
    return pd.DataFrame(rows)


def test_seed_replication_does_not_shrink_the_interval():
    """The core pseudo-replication guard: duplicating identical seeds must not
    make the confidence interval narrower."""
    one = _frame({("s1", "a"): [10.0], ("s1", "b"): [0.0],
                  ("s2", "a"): [20.0], ("s2", "b"): [5.0],
                  ("s3", "a"): [8.0], ("s3", "b"): [1.0]})
    five = _frame({("s1", "a"): [10.0] * 5, ("s1", "b"): [0.0] * 5,
                   ("s2", "a"): [20.0] * 5, ("s2", "b"): [5.0] * 5,
                   ("s3", "a"): [8.0] * 5, ("s3", "b"): [1.0] * 5})
    r1 = configuration_level_paired(one, "ens_kwh", "a", "b")
    r5 = configuration_level_paired(five, "ens_kwh", "a", "b")
    assert r1.n_configurations == r5.n_configurations == 3
    assert r1.n_runs == 6 and r5.n_runs == 30
    assert abs(r1.mean_diff - r5.mean_diff) < 1e-9
    width1 = r1.ci_high - r1.ci_low
    width5 = r5.ci_high - r5.ci_low
    assert abs(width1 - width5) < 1e-6, "seed duplication must not narrow the CI"
    assert abs(r1.p_two_sided - r5.p_two_sided) < 1e-9


def test_two_sided_permutation_is_exact_for_small_n():
    df = _frame({("s1", "a"): [3.0], ("s1", "b"): [1.0],
                 ("s2", "a"): [4.0], ("s2", "b"): [1.0],
                 ("s3", "a"): [5.0], ("s3", "b"): [1.0]})
    r = configuration_level_paired(df, "ens_kwh", "a", "b")
    assert r.test.startswith("exact-sign-flip")
    # all three differences positive -> only the all-positive and all-negative
    # sign assignments reach the observed |mean|: p = 2/8
    assert abs(r.p_two_sided - 0.25) < 1e-9


def test_no_difference_gives_large_p_and_interval_containing_zero():
    df = _frame({f"s{i}": None for i in range(0)} or
                {("s1", "a"): [1.0], ("s1", "b"): [1.0],
                 ("s2", "a"): [2.0], ("s2", "b"): [2.0],
                 ("s3", "a"): [3.0], ("s3", "b"): [3.0]})
    r = configuration_level_paired(df, "ens_kwh", "a", "b")
    assert r.mean_diff == 0.0
    assert r.ci_low <= 0.0 <= r.ci_high
    assert r.p_two_sided == 1.0
    assert r.n_zero_diff_configs == 3


def test_hierarchical_bootstrap_reports_zero_within_variance():
    df = _frame({("s1", "a"): [10.0] * 3, ("s1", "b"): [0.0] * 3,
                 ("s2", "a"): [20.0] * 3, ("s2", "b"): [5.0] * 3})
    out = hierarchical_bootstrap(df, "ens_kwh", "a", "b", n_boot=500)
    assert out["n_configurations"] == 2
    assert out["n_runs"] == 12
    assert out["mean_within_configuration_variance"] == 0.0
    assert "deterministic" in out["note"]


def test_holm_is_monotone_and_corrects():
    out = holm([0.001, 0.02, 0.04])
    assert out[0]["p_adj"] <= out[1]["p_adj"] <= out[2]["p_adj"]
    assert abs(out[0]["p_adj"] - 0.003) < 1e-9
    assert out[0]["reject"]


def test_degenerate_mask_flags_no_signal_configurations():
    df = _frame({("live", "a"): [10.0], ("live", "b"): [0.0],
                 ("dead", "a"): [0.0], ("dead", "b"): [0.0]})
    m = degenerate_mask(df, "ens_kwh").set_index("scenario")
    assert not m.loc["live", "all_methods_identical"]
    assert m.loc["dead", "all_methods_identical"]
    assert m.loc["dead", "all_zero"]
