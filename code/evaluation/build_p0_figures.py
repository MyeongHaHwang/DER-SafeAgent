"""Render IJCIP P0 figures and LaTeX tables.

Inputs (under ``code/results/``):
    ijcip_external_benchmark/{metrics,coverage_risk}.csv
    ijcip_latency/latency_metrics.csv
    ijcip_ablation/ablation_sensitive_metrics.csv,
                     ablation_sensitive_by_scenario.csv
    ijcip_hitl_sensitivity/{hitl_triggered_metrics,
                              hitl_by_operator_behavior,
                              hitl_by_slo}.csv
    ijcip_adversarial_safety/{robustness_metrics_expanded,
                                family_breakdown_expanded}.csv
    ijcip_objective_sensitivity/{objective_grid_expanded,
                                   pareto_points_expanded,
                                   risk_posture_summary}.csv
    ijcip_opendss_smoke/opendss_smoke_metrics.csv

Outputs:
    paper/figure/ijcip_p0/*.{pdf,png}
    code/results/ijcip_tables/table_*.tex
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("code/results")
FIG_DIR = Path("paper/figure")
TBL_DIR = RESULTS / "ijcip_tables"


DETECTOR_ORDER = [
    "rule_ids", "single_llm", "prior_mas",
    "safe_single_llm", "deterministic_energy_policy",
    "single_llm_with_caution", "prior_mas_with_safety",
    "der_secagent",
]
DETECTOR_LABEL = {
    "rule_ids":                    "rule_ids",
    "single_llm":                  "single_llm",
    "prior_mas":                   "prior_mas",
    "safe_single_llm":             "safe_single_llm",
    "deterministic_energy_policy": "deterministic",
    "single_llm_with_caution":     "single_llm+caution",
    "prior_mas_with_safety":       "prior_mas+safety",
    "der_secagent":                "DER-SecAgent (ours)",
}
DETECTOR_COLOR = {
    "rule_ids":                    "#888888",
    "single_llm":                  "#e07a3a",
    "prior_mas":                   "#3b7dd8",
    "safe_single_llm":             "#a86bc1",
    "deterministic_energy_policy": "#cf843e",
    "single_llm_with_caution":     "#d34b6c",
    "prior_mas_with_safety":       "#5e9bd0",
    "der_secagent":                "#2e7a3a",
}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.{{pdf,png}}")


# ---------- figures ----------

def fig_external_coverage_risk():
    p = RESULTS / "ijcip_external_benchmark/coverage_risk.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for det, g in df.groupby("detector"):
        g = g.sort_values("threshold")
        ax.plot(g["coverage"], g["selective_risk"], marker="o",
                color=DETECTOR_COLOR.get(det, "#888"),
                label=DETECTOR_LABEL.get(det, det), linewidth=1.6)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Coverage--risk on UNSW-MG24 external benchmark")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "external_benchmark_coverage_risk")


def fig_latency_real_backend():
    p = RESULTS / "ijcip_latency/latency_metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    backends = df["backend"].unique().tolist()
    ks = sorted(df["k"].unique().tolist())
    x = np.arange(len(backends)); w = 0.35
    for i, k in enumerate(ks):
        sub = df[df["k"] == k].set_index("backend").reindex(backends)
        means = sub["mean_ms"].astype(float).fillna(0).values
        statuses = sub["backend_status"].fillna("?").values
        bars = ax.bar(x + (i - 0.5) * w, means, w, label=k,
                       edgecolor="black", linewidth=0.4)
        # annotate not_run as text on bar
        for b, m, s in zip(bars, means, statuses):
            if s != "ok":
                ax.text(b.get_x() + b.get_width() / 2, 1e-3,
                          "not run", rotation=90, fontsize=7,
                          ha="center", va="bottom", color="#888")
    ax.set_xticks(x); ax.set_xticklabels(backends, rotation=20, ha="right",
                                              fontsize=8)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_yscale("log")
    ax.set_title("LLM latency by backend × K (bars: ms; annotated 'not run' rows)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "latency_real_backend")


def fig_ablation_sensitive():
    p = RESULTS / "ijcip_ablation/ablation_sensitive_metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    metrics = ["action_accuracy", "correct_refusal", "safe_fallback",
                 "policy_violation"]
    colors  = ["#2e7a3a", "#3b7dd8", "#7bb38c", "#c0392b"]
    flags = df["ablation"].tolist()
    fig, ax = plt.subplots(figsize=(11, 4))
    n_m = len(metrics); w = 0.85 / n_m
    x = np.arange(len(flags))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        offset = (i - (n_m - 1) / 2) * w
        ax.bar(x + offset, df[m].values, w, color=c, edgecolor="black",
                linewidth=0.4, label=m)
    ax.set_xticks(x); ax.set_xticklabels(flags, rotation=25, ha="right",
                                              fontsize=8)
    ax.set_ylabel("rate (mean over 8 sensitive scenarios × 3 seeds)")
    ax.set_title("DER-SecAgent ablation × sensitive scenarios")
    ax.set_ylim(0, 1.10); ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "ablation_sensitive_metrics")


def fig_hitl_triggered():
    p_op = RESULTS / "ijcip_hitl_sensitivity/hitl_by_operator_behavior.csv"
    p_slo = RESULTS / "ijcip_hitl_sensitivity/hitl_by_slo.csv"
    if not p_op.exists() or not p_slo.exists(): return
    op = pd.read_csv(p_op); slo = pd.read_csv(p_slo)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    # Left: by operator
    ops = op["operator"].tolist()
    metrics = [("unsafe_command_rate",  "unsafe-command",  "#c0392b"),
                 ("fallback_action_rate", "safe-fallback",   "#2e7a3a"),
                 ("wrong_auto_execution", "wrong-auto-exec", "#d34b6c")]
    x = np.arange(len(ops)); w = 0.27
    for i, (col, lbl, c) in enumerate(metrics):
        axes[0].bar(x + (i - 1) * w, op[col].astype(float).values, w,
                      label=lbl, color=c, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ops, rotation=15,
                                                          ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("rate"); axes[0].set_title("By operator behaviour")
    axes[0].legend(fontsize=8); axes[0].grid(True, axis="y", alpha=0.3)

    # Right: by SLO --- ENS proxy + fallback rate as twin axes
    axes[1].plot(slo["slo_s"], slo["unsafe_command_rate"], marker="o",
                   color="#c0392b", linewidth=1.8, label="unsafe-command")
    axes[1].plot(slo["slo_s"], slo["fallback_action_rate"], marker="s",
                   color="#2e7a3a", linewidth=1.8, label="safe-fallback")
    axes[1].set_xlabel("SLO (s)"); axes[1].set_ylabel("rate")
    axes[1].set_ylim(0, 1.05); axes[1].set_title("By HITL SLO")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "hitl_triggered_sensitivity")


def fig_adversarial_expanded():
    p = RESULTS / "ijcip_adversarial_safety/family_breakdown_expanded.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    fams = sorted(df["perturbation"].unique().tolist())
    detectors = [d for d in DETECTOR_ORDER if d in df["detector"].unique()]
    fig, ax = plt.subplots(figsize=(15, 5))
    n_det = len(detectors)
    x = np.arange(len(fams))
    w = 0.85 / max(n_det, 1)
    for i, det in enumerate(detectors):
        sub = (df[df["detector"] == det].set_index("perturbation")
               .reindex(fams))
        offset = (i - (n_det - 1) / 2) * w
        ax.bar(x + offset, sub["policy_violation_rate"].astype(float).values,
                w, color=DETECTOR_COLOR.get(det, "#888"),
                label=DETECTOR_LABEL.get(det, det),
                edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=25, ha="right",
                                              fontsize=8)
    ax.set_ylabel("policy-violation rate (lower is better)")
    ax.set_title(f"Adversarial robustness across {len(fams)} perturbation families "
                  f"× {n_det} detectors (50 cases × 5 seeds)")
    ax.set_ylim(0, 1.10); ax.legend(fontsize=7, ncol=4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "adversarial_expanded_breakdown")


def fig_objective_expanded():
    p = RESULTS / "ijcip_objective_sensitivity/objective_grid_expanded.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    if df.empty: return

    # Heatmap of selected_action over (ens_weight, tier_penalty),
    # averaged over conf_threshold.
    actions = ["no_op", "freeze_setpoint", "throttle_ramp",
                 "request_ied_revalidation", "isolate_inverter"]
    color_map = {"no_op": "#cccccc", "freeze_setpoint": "#2e7a3a",
                   "throttle_ramp": "#7bb38c",
                   "request_ied_revalidation": "#3b7dd8",
                   "isolate_inverter": "#c0392b"}
    df_mode = (df.groupby(["ens_weight", "tier_penalty"], as_index=False)
                  ["selected_action"].agg(lambda s: s.mode().iloc[0]))
    pivot = df_mode.pivot(index="tier_penalty", columns="ens_weight",
                            values="selected_action")
    fig, ax = plt.subplots(figsize=(7, 4))
    rows, cols = pivot.shape
    for i in range(rows):
        for j in range(cols):
            a = pivot.iloc[i, j]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          facecolor=color_map.get(a, "#fff"),
                                          edgecolor="black", linewidth=0.4))
            ax.text(j, i, a, ha="center", va="center", fontsize=7,
                     color="white" if a in {"isolate_inverter", "freeze_setpoint",
                                              "request_ied_revalidation"} else "black")
    ax.set_xticks(range(cols))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(rows))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    ax.set_xlabel("ENS weight"); ax.set_ylabel("Tier penalty")
    ax.set_xlim(-0.5, cols - 0.5); ax.set_ylim(-0.5, rows - 0.5)
    ax.set_title("Coordinator-selected action over the objective grid "
                  "(trade-off scenario)")
    fig.tight_layout()
    _save(fig, "objective_sensitivity_expanded")


def fig_opendss_smoke():
    p = RESULTS / "ijcip_opendss_smoke/opendss_smoke_metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    df_ok = df[df["status"] == "ok"]
    if df_ok.empty: return
    agg = (df_ok.groupby("detector", as_index=False)
                  [["ens_kwh", "curt_kwh", "voltage_min_pu",
                     "voltage_max_pu"]].mean())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    detectors = agg["detector"].tolist()
    x = np.arange(len(detectors))

    axes[0].bar(x, agg["ens_kwh"], color="#c0392b", edgecolor="black",
                  linewidth=0.4)
    axes[0].bar(x, agg["curt_kwh"], color="#3b7dd8", edgecolor="black",
                  linewidth=0.4, alpha=0.6, bottom=agg["ens_kwh"])
    axes[0].set_xticks(x); axes[0].set_xticklabels(detectors, rotation=15,
                                                          ha="right", fontsize=8)
    axes[0].set_ylabel("kWh"); axes[0].set_title("OpenDSS: ENS (red) + curtailment (blue)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].errorbar(x, agg["voltage_min_pu"],
                       yerr=[agg["voltage_min_pu"] * 0,
                              agg["voltage_max_pu"] - agg["voltage_min_pu"]],
                       fmt="o", color="#2e7a3a", capsize=5, linewidth=1.6,
                       label="bus voltage range")
    axes[1].axhline(0.95, color="#888", linestyle=":", linewidth=0.8,
                       label="ANSI C84.1 lower")
    axes[1].set_xticks(x); axes[1].set_xticklabels(detectors, rotation=15,
                                                          ha="right", fontsize=8)
    axes[1].set_ylabel("Voltage (pu)")
    axes[1].set_title("OpenDSS: bus voltage range")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "opendss_smoke_validation")


# ---------- LaTeX tables ----------

def _tex_escape(s: str) -> str:
    """Escape LaTeX special characters that appear in CSV-derived
    identifiers (detector / scenario / status names)."""
    if s is None:
        return ""
    out = str(s)
    for c in ("\\&", "\\#", "\\$", "\\%"):
        pass  # placeholder; run actual replacements below
    out = (out.replace("\\", r"\textbackslash{}")
              .replace("&", r"\&").replace("#", r"\#")
              .replace("$", r"\$").replace("%", r"\%")
              .replace("_", r"\_"))
    return out


def _write_table(name: str, content: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    (TBL_DIR / name).write_text(content + "\n")
    print(f"  wrote {TBL_DIR / name}")


def table_external_benchmark():
    p = RESULTS / "ijcip_external_benchmark/metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    # Submission-grade view: only datasets we actually ran. Adapters for
    # other public benchmarks (SWaT, WADI, EPIC) are kept in the
    # repository for reproducibility but are dropped from the paper
    # table since they would otherwise carry only ``not_run`` rows.
    df = df[df.get("status", "ok") != "not_run"].copy()
    rows = ["\\begin{tabular}{llrrrr}", "\\toprule",
              "Dataset & Detector & N & Acc. & macro-F1 "
              "& abstain \\\\",
              "\\midrule"]
    det_order = ["rule_ids", "single_llm", "der_secagent"]
    df["det_rank"] = df["detector"].map(
        {d: i for i, d in enumerate(det_order)}).fillna(99)
    df = df.sort_values(["dataset", "det_rank"]).drop(columns="det_rank")
    for _, r in df.iterrows():
        ds = _tex_escape(r["dataset"])
        det = _tex_escape(r["detector"])
        rows.append(
            f"{ds} & {det} "
            f"& {int(r['n_samples'])} & {r['accuracy']:.3f} "
            f"& {r['macro_f1']:.3f} & {r['abstain_rate']:.3f} \\\\"
        )
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_external_benchmark.tex", "\n".join(rows))


def table_latency():
    p = RESULTS / "ijcip_latency/latency_metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    rows = ["\\begin{tabular}{llrrrl}", "\\toprule",
              "Backend & K & N & mean (ms) & p95 (ms) & status \\\\",
              "\\midrule"]
    for _, r in df.iterrows():
        b = _tex_escape(r["backend"]); k = _tex_escape(r["k"])
        if r["backend_status"] == "ok":
            rows.append(
                f"{b} & {k} & {int(r['n_calls'])} "
                f"& {r['mean_ms']:.4f} & {r['p95_ms']:.4f} & ok \\\\")
        else:
            status = _tex_escape(r["backend_status"])
            reason = _tex_escape(str(r["not_run_reason"])[:60])
            rows.append(
                f"{b} & {k} & --- & --- & --- "
                f"& \\textit{{{status}: {reason}}} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_latency.tex", "\n".join(rows))


def table_ablation_sensitive():
    p = RESULTS / "ijcip_ablation/ablation_sensitive_metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
              "Ablation & action\\_acc & correct\\_refusal & safe\\_fallback "
              "& policy\\_viol. & ttm (ms) \\\\",
              "\\midrule"]
    for _, r in df.iterrows():
        rows.append(
            f"{_tex_escape(r['ablation'])} & {r['action_accuracy']:.3f} "
            f"& {r['correct_refusal']:.3f} & {r['safe_fallback']:.3f} "
            f"& {r['policy_violation']:.3f} "
            f"& {r['time_to_mitigation_ms']:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_ablation_sensitive.tex", "\n".join(rows))


def table_hitl_triggered():
    p = RESULTS / "ijcip_hitl_sensitivity/hitl_by_operator_behavior.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
              "Operator & escalation & wrong\\_auto & unsafe\\_cmd "
              "& fallback & SLO\\_expired \\\\",
              "\\midrule"]
    for _, r in df.iterrows():
        rows.append(
            f"{_tex_escape(r['operator'])} & {r['n_hitl_escalations']:.2f} "
            f"& {r['wrong_auto_execution']:.2f} "
            f"& {r['unsafe_command_rate']:.2f} "
            f"& {r['fallback_action_rate']:.2f} "
            f"& {r['n_slo_expired']:.2f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_hitl_triggered.tex", "\n".join(rows))


def table_adversarial_expanded():
    p = RESULTS / "ijcip_adversarial_safety/robustness_metrics_expanded.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    rows = ["\\begin{tabular}{lrrrrr}", "\\toprule",
              "Detector & policy\\_viol. & forbidden & safe\\_fallback "
              "& correct\\_refusal & schema\\_fail \\\\",
              "\\midrule"]
    for det in DETECTOR_ORDER:
        sub = df[df["detector"] == det]
        if sub.empty: continue
        r = sub.iloc[0]
        rows.append(
            f"{_tex_escape(DETECTOR_LABEL[det])} & {r['policy_violation_rate']:.3f} "
            f"& {r['forbidden_action_rate']:.3f} "
            f"& {r['safe_fallback_rate']:.3f} "
            f"& {r['correct_refusal_rate']:.3f} "
            f"& {r['schema_failure_rate']:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_adversarial_expanded.tex", "\n".join(rows))


def table_objective_sensitivity():
    p = RESULTS / "ijcip_objective_sensitivity/risk_posture_summary.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    rows = ["\\begin{tabular}{lrr}", "\\toprule",
              "Risk posture & grid points & fraction \\\\", "\\midrule"]
    for _, r in df.iterrows():
        rows.append(
            f"{_tex_escape(r['risk_posture'])} & {int(r['n_grid_points'])} "
            f"& {r['frac_grid']:.3f} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_objective_sensitivity.tex", "\n".join(rows))


def table_opendss_smoke():
    p = RESULTS / "ijcip_opendss_smoke/opendss_smoke_metrics.csv"
    if not p.exists(): return
    df = pd.read_csv(p)
    rows = ["\\begin{tabular}{llrrrrl}", "\\toprule",
              "Scenario & Detector & ENS & curt & v\\_min "
              "& v\\_max & action \\\\", "\\midrule"]
    # Aggregate seeds: identical OpenDSS solve gives identical numbers per
    # (scenario, detector); we display one row with the seed count baked
    # into the table caption rather than three duplicate rows.
    keep = df[df["status"] == "ok"].copy()
    grouped = keep.groupby(["scenario", "detector"], sort=False).agg({
        "ens_kwh": "mean", "curt_kwh": "mean",
        "voltage_min_pu": "min", "voltage_max_pu": "max",
        "selected_action": lambda s: s.mode().iloc[0] if not s.mode().empty else "no_op",
    }).reset_index()
    # honour the canonical detector ordering for readability
    det_order = ["rule_ids", "single_llm", "der_secagent"]
    grouped["det_rank"] = grouped["detector"].map(
        {d: i for i, d in enumerate(det_order)}).fillna(99)
    grouped = grouped.sort_values(["scenario", "det_rank"]).drop(columns="det_rank")
    for _, r in grouped.iterrows():
        scn = _tex_escape(r["scenario"]); det = _tex_escape(r["detector"])
        act = _tex_escape(r["selected_action"])
        rows.append(
            f"{scn} & {det} "
            f"& {r['ens_kwh']:.2f} & {r['curt_kwh']:.2f} "
            f"& {r['voltage_min_pu']:.3f} & {r['voltage_max_pu']:.3f} "
            f"& {act} \\\\")
    # Append a single "not_run" / fallback line per (scenario, detector) when
    # any seed for that pair failed.
    bad = df[df["status"] != "ok"].copy()
    for (scn, det), g in bad.groupby(["scenario", "detector"], sort=False):
        reason = _tex_escape(str(g["reason"].iloc[0])[:30])
        rows.append(
            f"{_tex_escape(scn)} & {_tex_escape(det)} "
            f"& --- & --- & --- & --- & \\textit{{{reason}}} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write_table("table_opendss_smoke.tex", "\n".join(rows))


# ---------- driver ----------

def main() -> None:
    fig_external_coverage_risk()
    fig_latency_real_backend()
    fig_ablation_sensitive()
    fig_hitl_triggered()
    fig_adversarial_expanded()
    fig_objective_expanded()
    fig_opendss_smoke()
    table_external_benchmark()
    table_latency()
    table_ablation_sensitive()
    table_hitl_triggered()
    table_adversarial_expanded()
    table_objective_sensitivity()
    table_opendss_smoke()
    print("done")


if __name__ == "__main__":
    main()
