"""Configuration-level and hierarchical statistical inference (revision R1-C3).

Replaces the published analysis, which resampled 15 (scenario, seed) pairs as
if independent and reported a one-sided p-value. Two facts force the change:

1. The harness is deterministic given a configuration — the five published
   seeds produce byte-identical physical outcomes (verified in Phase 2), so a
   seed is a repeated measurement of one configuration, not a new observation.
2. The unit an operator cares about is the scenario configuration (attack
   type/magnitude/duration x operating condition), not the RNG draw.

Provided here:

* ``configuration_level_paired`` — aggregate seeds within a configuration,
  then a two-sided paired permutation test (exact when 2**n is small enough,
  otherwise Monte Carlo) plus a paired bootstrap CI over configurations.
* ``hierarchical_bootstrap`` — resample configurations with replacement, then
  seeds within each selected configuration, for a variance estimate that
  respects the nesting.
* ``holm`` — Holm-Bonferroni over a declared family of comparisons.
* ``degenerate_mask`` — flags configurations that carry no signal (all methods
  identical, or zero variance across methods) for the sensitivity analysis.

Effect sizes: mean paired difference in native units (kWh) and a paired
Cohen's d_z. No universal-superiority language is produced by this module —
callers report results "within the evaluated configuration library".
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PairedResult:
    metric: str
    method_a: str
    method_b: str
    n_configurations: int
    n_runs: int
    mean_diff: float          # a - b (positive: a is worse when lower is better)
    ci_low: float
    ci_high: float
    cohens_dz: float
    p_two_sided: float
    test: str
    n_zero_diff_configs: int

    def to_row(self) -> dict:
        return asdict(self)


def aggregate_configurations(df: pd.DataFrame, metric: str,
                             config_key: str = "scenario",
                             method_key: str = "detector",
                             seed_key: str = "seed") -> pd.DataFrame:
    """Mean over seeds within (configuration, method). Also records how many
    runs backed each cell so n_runs and n_configurations stay distinguishable."""
    g = (df.groupby([config_key, method_key])[metric]
         .agg(["mean", "count", "std"]).reset_index()
         .rename(columns={"mean": metric, "count": "n_runs", "std": "seed_std"}))
    return g


def _paired_vectors(agg: pd.DataFrame, metric: str, a: str, b: str,
                    config_key: str, method_key: str):
    pa = agg[agg[method_key] == a].set_index(config_key)[metric]
    pb = agg[agg[method_key] == b].set_index(config_key)[metric]
    common = sorted(set(pa.index) & set(pb.index))
    return np.array([pa[c] for c in common]), np.array([pb[c] for c in common]), common


def _permutation_p(diff: np.ndarray, n_perm: int = 20000, seed: int = 0) -> tuple[float, str]:
    """Two-sided paired permutation (sign-flip) test on the mean difference."""
    obs = abs(diff.mean())
    n = len(diff)
    if n == 0:
        return float("nan"), "none"
    if n <= 20:                                    # exact enumeration
        count = 0
        total = 0
        for signs in itertools.product([1, -1], repeat=n):
            total += 1
            if abs((diff * np.array(signs)).mean()) >= obs - 1e-12:
                count += 1
        return count / total, f"exact-sign-flip(2^{n})"
    rng = np.random.default_rng(seed)
    signs = rng.choice([1, -1], size=(n_perm, n))
    stats = np.abs((signs * diff).mean(axis=1))
    return float(((stats >= obs - 1e-12).sum() + 1) / (n_perm + 1)), f"mc-sign-flip({n_perm})"


def configuration_level_paired(df: pd.DataFrame, metric: str, method_a: str,
                               method_b: str, config_key: str = "scenario",
                               method_key: str = "detector", seed_key: str = "seed",
                               n_boot: int = 10000, alpha: float = 0.05,
                               seed: int = 0) -> PairedResult:
    agg = aggregate_configurations(df, metric, config_key, method_key, seed_key)
    va, vb, configs = _paired_vectors(agg, metric, method_a, method_b,
                                      config_key, method_key)
    diff = va - vb
    n = len(diff)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) \
        if n else np.array([np.nan])
    sd = diff.std(ddof=1) if n > 1 else 0.0
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    p, test = _permutation_p(diff, seed=seed)
    n_runs = int(agg[agg[method_key].isin([method_a, method_b])]["n_runs"].sum())
    return PairedResult(
        metric=metric, method_a=method_a, method_b=method_b,
        n_configurations=n, n_runs=n_runs,
        mean_diff=float(diff.mean()) if n else float("nan"),
        ci_low=float(np.quantile(boots, alpha / 2)),
        ci_high=float(np.quantile(boots, 1 - alpha / 2)),
        cohens_dz=dz, p_two_sided=p, test=test,
        n_zero_diff_configs=int((np.abs(diff) < 1e-12).sum()),
    )


def hierarchical_bootstrap(df: pd.DataFrame, metric: str, method_a: str,
                           method_b: str, config_key: str = "scenario",
                           method_key: str = "detector", seed_key: str = "seed",
                           n_boot: int = 10000, alpha: float = 0.05,
                           seed: int = 0) -> dict:
    """Two-level resampling: configurations first, then seeds within each
    selected configuration. When the within-configuration variance is zero
    (deterministic harness), this reduces to the configuration-level bootstrap
    — which is the honest result, not an artefact."""
    rng = np.random.default_rng(seed)
    sub = df[df[method_key].isin([method_a, method_b])]
    configs = sorted(sub[config_key].unique())
    # per (config, method): the vector of per-seed values
    cell: dict[tuple[str, str], np.ndarray] = {}
    for (c, m), g in sub.groupby([config_key, method_key]):
        cell[(c, m)] = g[metric].to_numpy()

    usable = [c for c in configs if (c, method_a) in cell and (c, method_b) in cell]
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(usable, size=len(usable), replace=True)
        diffs = []
        for c in picked:
            va, vb = cell[(c, method_a)], cell[(c, method_b)]
            ia = rng.integers(0, len(va), len(va))
            ib = rng.integers(0, len(vb), len(vb))
            diffs.append(va[ia].mean() - vb[ib].mean())
        stats.append(np.mean(diffs))
    stats = np.array(stats)
    point = float(np.mean([cell[(c, method_a)].mean() - cell[(c, method_b)].mean()
                           for c in usable]))
    within_var = float(np.mean([cell[(c, m)].var() for c in usable
                                for m in (method_a, method_b)]))
    return {
        "metric": metric, "method_a": method_a, "method_b": method_b,
        "n_configurations": len(usable),
        "n_runs": int(sum(len(cell[(c, m)]) for c in usable
                          for m in (method_a, method_b))),
        "mean_diff": point,
        "ci_low": float(np.quantile(stats, alpha / 2)),
        "ci_high": float(np.quantile(stats, 1 - alpha / 2)),
        "bootstrap_se": float(stats.std(ddof=1)),
        "mean_within_configuration_variance": within_var,
        "note": ("within-configuration variance is zero: the harness is "
                 "deterministic given a configuration, so seeds contribute no "
                 "independent information" if within_var == 0.0 else ""),
        "n_boot": n_boot,
    }


def holm(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    out = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[i])
        running = max(running, adj)          # enforce monotonicity
        out[i] = {"p": p_values[i], "p_adj": running, "reject": running <= alpha}
    return out


def degenerate_mask(df: pd.DataFrame, metric: str, config_key: str = "scenario",
                    method_key: str = "detector") -> pd.DataFrame:
    """Flag configurations that cannot discriminate between methods."""
    rows = []
    for c, g in df.groupby(config_key):
        vals = g.groupby(method_key)[metric].mean()
        spread = float(vals.max() - vals.min())
        rows.append({
            config_key: c,
            "metric": metric,
            "n_methods": int(vals.size),
            "min": float(vals.min()), "max": float(vals.max()),
            "spread": spread,
            "all_methods_identical": spread < 1e-9,
            "all_zero": bool((vals.abs() < 1e-9).all()),
        })
    return pd.DataFrame(rows)
