"""Configuration-level statistical analysis for the revision (P0, R1-C3).

Unit of inference: the unique scenario configuration. Seeds are NOT treated as
independent observations — the harness is deterministic given a configuration
(verified in Phase 2), so a seed is a repeated measurement, and the analysis
reports n_configurations separately from n_runs.

Outputs (under code/results/<tag>/statistics/):
  hierarchical_bootstrap.csv         two-level (config -> seed) bootstrap
  configuration_level_effects.csv    paired two-sided permutation + bootstrap CI
  sensitivity_excluding_degenerate.csv  same, excluding non-discriminative configs
  statistical_report.md              narrative with the caveats stated

Run: python3 -m code.evaluation.stats.run_revision_stats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .hierarchical import (configuration_level_paired, degenerate_mask,
                           hierarchical_bootstrap, holm)

REVISION_TAG = "ijcip_revision_r1r2_20260805"
SUMMARY = Path("code/results") / REVISION_TAG / "llm_in_loop" / "llm_in_loop_summary.csv"
OUT = Path("code/results") / REVISION_TAG / "statistics"

REFERENCE = "der_secagent_heuristic_k3"
METRICS = ["ens_kwh", "curt_kwh"]


def _load() -> pd.DataFrame:
    """Load the per-(system, configuration) summary.

    The summary already carries a ``detector`` column contributed by the
    caution-metrics aggregate, so the system label is copied into a distinct
    ``method`` column rather than renamed over it.
    """
    df = pd.read_csv(SUMMARY)
    df["method"] = df["system"]
    if "seed" not in df.columns:
        df["seed"] = 0
    # keep one row per (method, configuration)
    df = df.drop_duplicates(subset=["method", "scenario", "seed"], keep="last")
    return df


def _family(df: pd.DataFrame, reference: str) -> list[str]:
    return [d for d in sorted(df.method.unique()) if d != reference]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=REFERENCE)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = _load()
    live = df[~df.is_calibration_control.astype(bool)] if "is_calibration_control" in df else df

    # ---- degenerate-configuration mask (shared by the sensitivity analysis)
    masks = []
    for m in METRICS:
        mk = degenerate_mask(live, m, method_key="method")
        masks.append(mk)
    mask = pd.concat(masks, ignore_index=True)
    mask.to_csv(OUT / "degenerate_configuration_mask.csv", index=False)

    cfg_rows, hier_rows, sens_rows = [], [], []
    for metric in METRICS:
        others = _family(live, args.reference)

        # ---- configuration-level paired analysis (primary)
        results = [configuration_level_paired(live, metric, o, args.reference, method_key="method")
                   for o in others]
        corrected = holm([r.p_two_sided for r in results])
        for r, c in zip(results, corrected):
            row = r.to_row()
            row["p_adj_holm"] = c["p_adj"]
            row["reject_at_0.05"] = c["reject"]
            row["family_size"] = len(results)
            cfg_rows.append(row)

        # ---- hierarchical bootstrap (secondary)
        for o in others:
            hier_rows.append(hierarchical_bootstrap(live, metric, o, args.reference, method_key="method"))

        # ---- sensitivity: drop configurations where every method is identical
        deg = mask[(mask.metric == metric) & mask.all_methods_identical].scenario.unique()
        keep = live[~live.scenario.isin(deg)]
        s_results = [configuration_level_paired(keep, metric, o, args.reference, method_key="method")
                     for o in others]
        s_corr = holm([r.p_two_sided for r in s_results])
        for r, c in zip(s_results, s_corr):
            row = r.to_row()
            row["p_adj_holm"] = c["p_adj"]
            row["reject_at_0.05"] = c["reject"]
            row["n_configurations_excluded"] = len(deg)
            row["excluded_configurations"] = ";".join(sorted(deg))
            sens_rows.append(row)

    pd.DataFrame(cfg_rows).to_csv(OUT / "configuration_level_effects.csv", index=False)
    pd.DataFrame(hier_rows).to_csv(OUT / "hierarchical_bootstrap.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(OUT / "sensitivity_excluding_degenerate.csv", index=False)

    # ---- narrative report
    cfg = pd.DataFrame(cfg_rows)
    sens = pd.DataFrame(sens_rows)
    n_cfg = int(live.scenario.nunique())
    n_runs = int(len(live))
    lines = [
        "# Statistical report — configuration-level inference",
        f"Revision tag: `{REVISION_TAG}`  |  Reference method: `{args.reference}`",
        "",
        "## Unit of inference",
        f"- Unique scenario configurations analysed: **{n_cfg}** "
        f"(calibration controls excluded from the primary analysis).",
        f"- Total runs backing them: **{n_runs}**.",
        "- The harness is deterministic given a configuration: repeated seeds "
        "reproduce a run byte-for-byte (verified in Phase 2). Seeds are therefore "
        "repeated measurements, not independent scenario-level observations, and "
        "the configuration is the unit of inference throughout.",
        "",
        "## Method",
        "- **Primary:** aggregate within configuration, then a two-sided paired "
        "sign-flip permutation test across configurations (exact enumeration when "
        "the number of configurations permits), with a paired bootstrap 95% CI "
        "and paired Cohen's $d_z$.",
        "- **Secondary:** hierarchical bootstrap resampling configurations first "
        "and seeds within each selected configuration.",
        "- Holm correction is applied across the declared family of method "
        "comparisons, per metric.",
        "- No one-sided test is used: no one-sided hypothesis was pre-registered "
        "before the results were seen. (The previously published analysis "
        "reported a one-sided p-value over 15 correlated scenario-seed pairs; "
        "that analysis is superseded here.)",
        "",
        "## Configuration-level effects (primary)",
        "",
        cfg.to_markdown(index=False) if len(cfg) else "_no rows_",
        "",
        "## Hierarchical bootstrap (secondary)",
        "",
        pd.DataFrame(hier_rows).to_markdown(index=False) if hier_rows else "_no rows_",
        "",
        "## Sensitivity analysis excluding degenerate configurations",
        "Configurations on which every method produces an identical outcome carry "
        "no discriminative information; the table below repeats the primary "
        "analysis with them removed.",
        "",
        sens.to_markdown(index=False) if len(sens) else "_no rows_",
        "",
        "## Interpretation caveats",
        "- All statements hold **within the evaluated configuration library** and "
        "**across the tested operating and attack conditions**; they are not "
        "claims of universal superiority across DER deployments or attack "
        "distributions.",
        "- Confidence intervals quantify sampling uncertainty over the "
        "configuration library actually evaluated. A different library would give "
        "different intervals.",
        "- Where a contrast has many zero-difference configurations "
        "(`n_zero_diff_configs`), the effect is carried by a small number of "
        "configurations and should be read as such.",
    ]
    (OUT / "statistical_report.md").write_text("\n".join(lines))
    print(f"statistics -> {OUT} ({len(cfg)} configuration-level contrasts)")


if __name__ == "__main__":
    main()
