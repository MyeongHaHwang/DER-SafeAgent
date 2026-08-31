"""P1-D: configuration-level statistics for the final-architecture experiments.

Inference unit: UNIQUE CONFIGURATION. For P1-B (3 genuine stochastic episodes
per configuration) the per-configuration value is the episode mean, and a
hierarchical (two-stage) bootstrap over configurations then episodes provides
the interval; episodes are never pooled as independent configurations.

Tests: paired two-sided sign-flip permutation tests on per-configuration
differences; paired bootstrap 95% CIs; Cohen's d_z effect size; Holm
correction within each planned family.

Planned families:
  F1 (P1-B physical outcome): each arm vs D0 on ens_kwh and curt_kwh.
  F2 (P1-A OpenDSS): each LLM system vs OD0 on curt_kwh and
     voltage_violation_frac (ENS reported descriptively when zero-variance).
  F3 (P0-B gate): G1 vs G0 benign false-action (paired by configuration,
     McNemar-style sign-flip on the indicator).

Sensitivity: excluding zero-variance configurations and the benign control.

Run: python3 -m code.evaluation.final_safeagent.run_final_stats
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

TAG = "ijcip_final_safeagent_20260810"
RES = Path("code/results") / TAG
OUT = RES / "statistics"
N_PERM = 20000
N_BOOT = 10000
RNG = np.random.default_rng(20260810)


def signflip_p(diff: np.ndarray) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(diff) != 0."""
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    obs = abs(diff.mean())
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1))


def paired_boot_ci(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cohens_dz(diff: np.ndarray) -> float:
    diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def hier_boot_ci(sub: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Two-stage bootstrap: resample configurations, then episodes within."""
    groups = [g[metric].to_numpy() for _, g in sub.groupby("scenario")]
    means = []
    for _ in range(2000):
        pick = RNG.integers(0, len(groups), size=len(groups))
        vals = [RNG.choice(groups[i], size=len(groups[i]), replace=True).mean()
                for i in pick]
        means.append(np.mean(vals))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def family_f1() -> pd.DataFrame:
    df = pd.read_csv(RES / "p1b_projection" / "p1b_raw.csv")
    per_cfg = (df.groupby(["arm", "scenario"])
                 [["ens_kwh", "curt_kwh"]].mean().reset_index())
    d0 = per_cfg[per_cfg.arm == "D0"].set_index("scenario")
    rows, pvals = [], []
    arms = [a for a in per_cfg.arm.unique() if a != "D0"]
    for arm, metric in itertools.product(arms, ("ens_kwh", "curt_kwh")):
        a = per_cfg[per_cfg.arm == arm].set_index("scenario")
        common = a.index.intersection(d0.index)
        diff = (a.loc[common, metric] - d0.loc[common, metric]).to_numpy()
        lo, hi = paired_boot_ci(diff)
        p = signflip_p(diff)
        # sensitivity: drop configurations where every arm ties (zero variance)
        nz = diff[np.abs(diff) > 1e-9]
        rows.append({"family": "F1", "contrast": f"{arm} - D0",
                     "metric": metric, "n_configurations": len(common),
                     "n_total_runs": int(len(df[df.arm == arm])),
                     "mean_diff": float(np.nanmean(diff)),
                     "ci95_lo": lo, "ci95_hi": hi, "d_z": cohens_dz(diff),
                     "p_raw": p,
                     "n_nonzero_diff_configs": int(len(nz)),
                     "mean_diff_nonzero_only": (float(nz.mean()) if len(nz)
                                                else 0.0)})
        pvals.append(p)
    adj = holm(pvals)
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    return pd.DataFrame(rows)


def family_f2() -> pd.DataFrame:
    df = pd.read_csv(RES / "p1a_opendss" / "p1a_raw.csv")
    df = df[~df.is_control]
    per = df.set_index(["system", "configuration_id"])
    d0 = df[df.system == "OD0"].set_index("configuration_id")
    rows, pvals = [], []
    for system in sorted(set(df.system) - {"OD0"}):
        a = df[df.system == system].set_index("configuration_id")
        common = a.index.intersection(d0.index)
        for metric in ("curt_kwh", "voltage_violation_frac"):
            diff = (a.loc[common, metric] - d0.loc[common, metric]).to_numpy(float)
            lo, hi = paired_boot_ci(diff)
            p = signflip_p(diff)
            rows.append({"family": "F2", "contrast": f"{system} - OD0",
                         "metric": metric, "n_configurations": len(common),
                         "mean_diff": float(np.nanmean(diff)),
                         "ci95_lo": lo, "ci95_hi": hi,
                         "d_z": cohens_dz(diff), "p_raw": p})
            pvals.append(p)
        ens = a.loc[common, "ens_kwh"]
        rows.append({"family": "F2", "contrast": f"{system} - OD0",
                     "metric": "ens_kwh(descriptive)",
                     "n_configurations": len(common),
                     "mean_diff": float((ens - d0.loc[common, "ens_kwh"]).mean()),
                     "ci95_lo": float("nan"), "ci95_hi": float("nan"),
                     "d_z": float("nan"), "p_raw": float("nan")})
    adj = holm([p for p in pvals])
    it = iter(adj)
    for r in rows:
        if not np.isnan(r["p_raw"]):
            r["p_holm"] = next(it)
        else:
            r["p_holm"] = float("nan")
    return pd.DataFrame(rows)


def family_f3() -> pd.DataFrame:
    df = pd.read_csv(RES / "p0b_gate" / "gate_test_raw.csv")
    rows = []
    for base in ("D0", "Q1"):
        g0 = df[(df.system == f"{base}-G0")
                & df.condition_kind.str.startswith("benign")]
        g1 = df[(df.system == f"{base}-G1")
                & df.condition_kind.str.startswith("benign")]
        m = g0.merge(g1, on="scenario", suffixes=("_g0", "_g1"))
        diff = ((m.n_actions_g1 > 0).astype(float)
                - (m.n_actions_g0 > 0).astype(float)).to_numpy()
        lo, hi = paired_boot_ci(diff)
        rows.append({"family": "F3",
                     "contrast": f"{base}: G1 - G0 benign false-action",
                     "metric": "false_action_indicator",
                     "n_configurations": len(m),
                     "mean_diff": float(diff.mean()), "ci95_lo": lo,
                     "ci95_hi": hi, "d_z": cohens_dz(diff),
                     "p_raw": signflip_p(diff)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(list(out.p_raw))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for fn in (family_f1, family_f2, family_f3):
        try:
            frames.append(fn())
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    res = pd.concat(frames, ignore_index=True)
    res.round(6).to_csv(OUT / "final_stats.csv", index=False)

    # hierarchical bootstrap for the P1-B primary metric, per arm
    try:
        df = pd.read_csv(RES / "p1b_projection" / "p1b_raw.csv")
        hier = {}
        for arm in df.arm.unique():
            sub = df[df.arm == arm]
            lo, hi = hier_boot_ci(sub, "ens_kwh")
            hier[arm] = {"ens_mean": float(sub.ens_kwh.mean()),
                         "hier_ci95": [lo, hi],
                         "n_configurations": int(sub.scenario.nunique()),
                         "n_total_runs": int(len(sub))}
        (OUT / "p1b_hierarchical.json").write_text(json.dumps(hier, indent=2))
    except FileNotFoundError:
        pass
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
