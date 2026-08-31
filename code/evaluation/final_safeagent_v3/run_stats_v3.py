"""V3 final statistics — implements ANALYSIS_PLAN_v3.md.

Configuration-level inference. Continuous outcomes: paired sign-flip
permutation + paired bootstrap CI + Cohen's d_z. Binary paired outcomes:
exact McNemar + Clopper-Pearson. Holm within each hypothesis family.
Consumes the v3 holdout (holdout_e2e_raw.csv) and adversarial
(adversarial_summary_*.csv) when present; skips families whose inputs are not
yet available and says so (no fabrication).

Run: python3 -m code.evaluation.final_safeagent_v3.run_stats_v3
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("code/results/ijcip_final_v3")
OUT = RES / "statistics"
SEED = 20260811
N_PERM = 20000
N_BOOT = 10000

# Common random numbers: the sign-flip and bootstrap resampling draws are a
# function of the sample size n only (seeded from SEED), and are shared across
# every contrast in a family. Identical paired-difference vectors therefore
# receive identical p-values and identical bootstrap CIs regardless of the
# order in which contrasts are evaluated. (The original 2026-08-11 run drew
# from one sequential RNG stream, so the byte-identical ModelOnly-Q/L vectors
# received different bootstrap samples and near-identical-but-unequal CIs.)
_SIGNS_CACHE: dict = {}
_BOOT_CACHE: dict = {}


def _signs_for(n):
    if n not in _SIGNS_CACHE:
        rng = np.random.default_rng([SEED, 1, n])
        _SIGNS_CACHE[n] = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    return _SIGNS_CACHE[n]


def _boot_idx_for(n):
    if n not in _BOOT_CACHE:
        rng = np.random.default_rng([SEED, 2, n])
        _BOOT_CACHE[n] = rng.integers(0, n, size=(N_BOOT, n))
    return _BOOT_CACHE[n]


def signflip_p(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    obs = abs(diff.mean())
    null = np.abs((_signs_for(len(diff)) * diff).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1))


def boot_ci(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    m = diff[_boot_idx_for(len(diff))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def dz(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    return float(diff.mean() / sd) if sd > 0 else 0.0


def mcnemar_exact(b, c):
    """Exact McNemar on discordant pairs (b, c)."""
    from scipy import stats
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return float(min(p, 1.0))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def family_f1_physical():
    p = RES / "holdout_e2e" / "holdout_e2e_raw.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    per = (df.groupby(["arm", "scenario", "kind"])[["ens_kwh", "curt_kwh"]]
           .mean().reset_index())
    atk = per[per.kind == "attack"]
    d0 = atk[atk.arm == "D0"].set_index("scenario")
    rows, pv = [], []
    for arm in [a for a in atk.arm.unique() if a != "D0"]:
        a = atk[atk.arm == arm].set_index("scenario")
        common = a.index.intersection(d0.index)
        for metric in ("ens_kwh", "curt_kwh"):
            diff = (a.loc[common, metric] - d0.loc[common, metric]).to_numpy()
            lo, hi = boot_ci(diff)
            pp = signflip_p(diff)
            rows.append({"family": "F1", "contrast": f"{arm}-D0",
                         "metric": metric, "n_cfg": len(common),
                         "mean_diff": float(np.nanmean(diff)),
                         "ci_lo": lo, "ci_hi": hi, "d_z": dz(diff),
                         "n_zero_diff": int(np.sum(np.abs(diff) < 1e-9)),
                         "p_raw": pp})
            pv.append(pp)
    adj = holm(pv)
    for r, a in zip(rows, adj):
        r["p_holm"] = a
    return pd.DataFrame(rows)


def family_f3_gate():
    p = RES / "holdout_e2e" / "holdout_e2e_raw.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    per = (df.groupby(["arm", "scenario", "kind"])["n_actions"]
           .mean().reset_index())
    ben = per[per.kind == "benign"]
    rows = []
    # exact McNemar: benign false-action D0 vs OPROJ (paired by scenario)
    for a1, a2 in [("D0", "OPROJ")]:
        m = (ben[ben.arm == a1].set_index("scenario").n_actions > 0).astype(int)
        n = (ben[ben.arm == a2].set_index("scenario").n_actions > 0).astype(int)
        common = m.index.intersection(n.index)
        b = int(((m.loc[common] == 1) & (n.loc[common] == 0)).sum())
        c = int(((m.loc[common] == 0) & (n.loc[common] == 1)).sum())
        rows.append({"family": "F3", "contrast": f"benign FA {a1} vs {a2}",
                     "metric": "false_action", "n_cfg": len(common),
                     "discordant_b": b, "discordant_c": c,
                     "p_raw": mcnemar_exact(b, c)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(list(out.p_raw))
    return out


def family_f2_adversarial():
    rows = []
    for be in ("qwen", "llama"):
        p = RES / "adversarial" / f"adversarial_summary_{be}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p)
        for _, r in s.iterrows():
            rows.append({"family": "F2", "backend": be, "arm": r["arm"],
                         "n": int(r["n"]), "irrev_executed": int(r["irrev_executed"]),
                         "irrev_exec_rate": float(r["irrev_exec_rate"]),
                         "irrev_exec_upper95": float(r["irrev_exec_upper95"]),
                         "policy_violation_rate": float(r["policy_violation_rate"])})
    return pd.DataFrame(rows) if rows else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, fn in (("F1_physical", family_f1_physical),
                     ("F2_adversarial", family_f2_adversarial),
                     ("F3_gate", family_f3_gate)):
        df = fn()
        if df is None:
            status[name] = "PENDING (inputs not yet available)"
            print(f"[stats] {name}: PENDING")
        else:
            df.round(6).to_csv(OUT / f"{name}.csv", index=False)
            status[name] = f"done ({len(df)} rows)"
            print(f"[stats] {name}:\n{df.round(4).to_string(index=False)}")
    (OUT / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
