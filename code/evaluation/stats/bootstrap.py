"""Paired bootstrap confidence intervals for method-vs-method comparisons.

We resample (scenario, seed) pairs with replacement and recompute the per-method
mean of the chosen metric. The point estimate is the difference of means; the
CI is the empirical percentile interval. Reports a one-sided p-value as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    method_a: str,
    method_b: str,
    pair_keys: tuple[str, ...] = ("scenario", "seed"),
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Returns Δ = mean(method_a) − mean(method_b) with bootstrap CI.

    Negative Δ favors method_a if `metric` is a cost (lower-is-better).
    """
    pivot = (df.pivot_table(index=list(pair_keys), columns="detector",
                            values=metric, aggfunc="mean")
               .dropna(subset=[method_a, method_b]))
    a = pivot[method_a].to_numpy()
    b = pivot[method_b].to_numpy()
    diffs = a - b
    point = float(diffs.mean())

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(diffs)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = diffs[idx].mean()

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    p_one_sided = float((boots >= 0).mean()) if point < 0 else float((boots <= 0).mean())

    return {
        "metric": metric,
        "a": method_a,
        "b": method_b,
        "n_pairs": int(n),
        "point_estimate": point,
        "ci_low": lo,
        "ci_high": hi,
        "p_one_sided": p_one_sided,
    }
