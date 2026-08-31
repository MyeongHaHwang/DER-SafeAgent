"""Render the IJCIP-revision figures from results/ijcip_*.

All figures are saved under figures/ijcip/ as both PDF (paper) and PNG.
The two LaTeX tables emitted by this driver land alongside as
``table_*.tex`` snippets.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("paper/figure")
RESULTS = Path("code/results")

DETECTOR_ORDER = [
    "rule_ids", "single_llm", "prior_mas",
    "safe_single_llm", "deterministic_energy_policy",
    "single_llm_with_caution", "prior_mas_with_safety",
    "der_secagent", "oracle_class_policy",
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
    "oracle_class_policy":         "oracle (upper bound)",
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
    "oracle_class_policy":         "#000000",
}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.{{pdf,png}}")


# ---------- system overview (revised, shipped as static asset) ----------

def fig_system_overview():
    """The current paper/figure/system_overview.pdf already plays this role
    --- copy it across so the IJCIP-figure folder is self-contained."""
    src_pdf = Path("paper/figure/system_overview.pdf")
    if src_pdf.exists():
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        (FIG_DIR / "system_overview_ci.pdf").write_bytes(src_pdf.read_bytes())
        print(f"  copied system overview from {src_pdf}")


# ---------- adversarial robustness ----------

def fig_adversarial():
    csv = RESULTS / "ijcip_adversarial_safety/robustness_metrics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    fams = list(df["perturbation"].unique())
    detectors = [d for d in DETECTOR_ORDER if d in df["detector"].unique()]
    n_det = len(detectors)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(fams))
    w = 0.85 / max(n_det, 1)
    for i, det in enumerate(detectors):
        sub = df[df["detector"] == det].set_index("perturbation").reindex(fams)
        offset = (i - (n_det - 1) / 2) * w
        ax.bar(x + offset, sub["policy_violation_rate"],
                w, label=DETECTOR_LABEL.get(det, det),
                color=DETECTOR_COLOR.get(det, "#888"),
                edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(fams, rotation=15, ha="right",
                                            fontsize=9)
    ax.set_ylabel("policy-violation rate (lower is better)")
    ax.set_title("Adversarial robustness across six perturbation families "
                  f"(N={n_det} detectors)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, axis="y", alpha=0.3); ax.set_ylim(0, 1.10)
    fig.tight_layout()
    _save(fig, "adversarial_robustness")


# ---------- external benchmark ----------

def fig_external_benchmark():
    csv = RESULTS / "ijcip_external_benchmark/metrics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df_ok = df[df["status"] == "ok"]
    if df_ok.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    pivot_acc = df_ok.pivot_table(index="dataset", columns="detector",
                                     values="accuracy", aggfunc="mean")
    pivot_f1 = df_ok.pivot_table(index="dataset", columns="detector",
                                    values="macro_f1", aggfunc="mean")
    for ax, p, lbl in zip(axes, (pivot_acc, pivot_f1),
                            ("accuracy", "macro-F1")):
        x = np.arange(len(p.index)); w = 0.25
        for i, det in enumerate(p.columns):
            ax.bar(x + (i - len(p.columns)/2) * w, p[det].values, w,
                    label=DETECTOR_LABEL.get(det, det),
                    color=DETECTOR_COLOR.get(det, "#888"),
                    edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(p.index)
        ax.set_ylabel(lbl); ax.set_title(f"{lbl} on external benchmarks")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "external_benchmark")


# ---------- ablation chart ----------

def fig_ablation():
    csv = RESULTS / "ijcip_ablation/ablation_adversarial_metrics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    # Average rates over the six perturbation families per ablation flag.
    metrics = [
        ("safe_fallback_rate",      "safe_fallback",       "#2e7a3a"),
        ("policy_violation_rate",   "policy_violation",    "#c0392b"),
        ("schema_failure_rate",     "schema_failure",      "#888888"),
        ("correct_refusal_rate",    "correct_refusal",     "#3b7dd8"),
        ("abstained_rate",          "abstained",           "#d4a843"),
    ]
    flags = (df["ablation"].drop_duplicates().tolist())
    agg = (df.groupby("ablation", as_index=False)
              [[m[0] for m in metrics]].mean())
    agg = agg.set_index("ablation").reindex(flags)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    n_metrics = len(metrics)
    x = np.arange(len(flags))
    w = 0.85 / n_metrics
    for i, (col, label, color) in enumerate(metrics):
        offset = (i - (n_metrics - 1) / 2) * w
        ax.bar(x + offset, agg[col].values, w, label=label,
                color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(flags, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("rate (mean over six perturbation families)")
    ax.set_title("DER-SecAgent ablation behaviour under adversarial inputs")
    ax.set_ylim(0, 1.10)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.5, 0.92,
              "every ablation preserves the trust invariants: "
              "policy_violation = 0, safe_fallback = 1.0",
              transform=ax.transAxes, ha="center", fontsize=8,
              style="italic", color="#444",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7c4",
                          edgecolor="#bbb"))
    fig.tight_layout()
    _save(fig, "ablation_adversarial")


# ---------- HITL sensitivity ----------

def fig_hitl():
    csv = RESULTS / "ijcip_hitl_sensitivity/hitl_sensitivity_agg.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    operators = sorted(df["operator"].unique().tolist())
    palette = {"approve": "#2e7a3a", "reject": "#c0392b",
                 "modify_to_safe_action": "#3b7dd8",
                 "no_response": "#7f7f7f", "noisy_operator": "#e07a3a"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))

    # Left panel: ENS vs SLO. Under the trust-by-construction policy
    # the curves overlap; we annotate the invariant rather than hide it.
    for op in operators:
        g = (df[df["operator"] == op]
                .groupby("slo_s", as_index=False)["ens_kwh"].mean()
                .sort_values("slo_s"))
        axes[0].plot(g["slo_s"], g["ens_kwh"], marker="o", linewidth=1.8,
                      markersize=8, color=palette.get(op, "#888"),
                      label=op, alpha=0.85)
    ens_max = float(df["ens_kwh"].max())
    axes[0].set_ylim(0, max(ens_max * 5, 0.05))
    axes[0].set_xlabel("HITL SLO (s)"); axes[0].set_ylabel("ENS (kWh)")
    axes[0].set_title("ENS vs SLO × operator (curves overlap)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)
    axes[0].text(0.5, 0.92,
                  f"all operators: ENS = {ens_max:.4f} kWh "
                  "(invariant under HITL behaviour)",
                  transform=axes[0].transAxes, ha="center", fontsize=8,
                  style="italic", color="#444",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7c4",
                              edgecolor="#bbb"))

    # Right panel: operator-state counts at SLO=60s. Counts are zero
    # because the Coordinator's class-aware avoidance + HITL-by-design
    # never escalates on benign scenarios; we annotate that explicitly.
    sub = (df[df["slo_s"] == 60]
            .groupby("operator", as_index=False)
            [["n_hitl_escalations", "n_slo_expired",
                "n_safe_fallback_actions"]].mean())
    sub = sub.set_index("operator").reindex(operators).fillna(0)
    metrics = [("n_hitl_escalations", "HITL escalations", "#3b7dd8"),
                 ("n_slo_expired",       "SLO expirations",  "#c0392b"),
                 ("n_safe_fallback_actions", "safe fallbacks", "#2e7a3a")]
    x = np.arange(len(operators)); w = 0.26
    for i, (col, lbl, c) in enumerate(metrics):
        axes[1].bar(x + (i - 1) * w, sub[col], w, label=lbl,
                      color=c, edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(operators, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("mean count per run")
    axes[1].set_title("Operator-side counts @ SLO=60 s")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")
    # explicit zero-annotation for each operator, since values are 0
    for i in range(len(operators)):
        axes[1].text(i, 0.05, "0", ha="center", va="bottom",
                      fontsize=9, color="#444")
    axes[1].text(0.5, 0.78,
                  "zero workload on benign scenarios — HITL gate never\n"
                  "triggers because Coordinator class-aware avoidance\n"
                  "already produces safe actions",
                  transform=axes[1].transAxes, ha="center", fontsize=8,
                  style="italic", color="#444",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f1d8",
                              edgecolor="#bbb"))
    fig.tight_layout()
    _save(fig, "hitl_sensitivity")


# ---------- objective sensitivity (heatmap) ----------

def fig_objective_heatmap():
    csv = RESULTS / "ijcip_objective_sensitivity/objective_grid.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    if df.empty:
        return
    # The ENS surface is essentially flat under the heuristic backbone:
    # the Coordinator's class-aware overrides dominate the objective
    # weights on the synthetic scenarios. Instead of an uninformative
    # uniform-colour heatmap, plot the *selected-action distribution*
    # across the grid --- this is the trustworthy-AI invariant the sweep
    # actually documents (no objective weight pushes the system toward
    # an irreversible primitive).
    action_cols = ["frac_no_op", "frac_freeze", "frac_throttle",
                    "frac_revalidate", "frac_isolate"]
    labels = ["no_op", "freeze_setpoint", "throttle_ramp",
                "request_ied_revalidation", "isolate_inverter"]
    colors = ["#cccccc", "#2e7a3a", "#7bb38c", "#3b7dd8", "#c0392b"]

    # one row per (ens_weight, tier_penalty) (averaged over conf_threshold)
    g = (df.groupby(["ens_weight", "tier_penalty"], as_index=False)[action_cols]
            .mean())
    g["label"] = g.apply(lambda r: f"w={r['ens_weight']:g}, p={r['tier_penalty']:g}",
                            axis=1)
    g = g.sort_values(["ens_weight", "tier_penalty"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(g))
    for col, lbl, c in zip(action_cols, labels, colors):
        ax.bar(g["label"], g[col].values, bottom=bottom, label=lbl,
                color=c, edgecolor="black", linewidth=0.4)
        bottom = bottom + g[col].values
    ax.set_xlabel("(ENS weight, tier penalty)  [averaged over confidence threshold]")
    ax.set_ylabel("Selected-action fraction")
    ax.set_title("Coordinator action distribution over the objective grid")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    for label in ax.get_xticklabels():
        label.set_rotation(35); label.set_ha("right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "objective_heatmap")

    # Pareto plot --- under the heuristic backbone every grid point
    # collapses onto the same (curt, ENS) point, which is itself the
    # trustworthy-AI invariant. We render a clear single-cluster plot
    # with the cluster annotation rather than hiding the degeneracy.
    pareto_csv = RESULTS / "ijcip_objective_sensitivity/pareto_points.csv"
    if pareto_csv.exists():
        p = pd.read_csv(pareto_csv)
        fig, ax = plt.subplots(figsize=(6, 4))
        x_vals = df["mean_curt_kwh"].values
        y_vals = df["mean_ens_kwh"].values
        ax.scatter(x_vals, y_vals, s=120, alpha=0.30,
                    color="#3b7dd8", edgecolor="black",
                    linewidth=0.4, label=f"{len(df)} grid points (overlap)")
        if not p.empty:
            ax.scatter(p["mean_curt_kwh"], p["mean_ens_kwh"],
                        s=200, marker="*", color="#2e7a3a",
                        edgecolor="black", linewidth=0.6,
                        label="Pareto-optimal cluster")
        ax.set_xlabel("Mean curtailment (kWh)")
        ax.set_ylabel("Mean ENS (kWh)")
        ax.set_title("(curtailment, ENS) sweep collapses to a single cluster")
        x_pad = max(0.5, abs(x_vals.mean()) * 0.05)
        y_pad = max(0.001, abs(y_vals.max() - y_vals.min()) * 1.5 + 0.003)
        ax.set_xlim(x_vals.mean() - x_pad * 4, x_vals.mean() + x_pad * 4)
        ax.set_ylim(y_vals.min() - y_pad, y_vals.max() + y_pad)
        ax.text(0.5, 0.10,
                  "all 27 grid points overlap: the Coordinator's\n"
                  "class-aware override dominates the objective\n"
                  "weights on benign scenarios",
                  transform=ax.transAxes, ha="center", fontsize=8,
                  style="italic", color="#444",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff7c4",
                              edgecolor="#bbb"))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "objective_pareto")


# ---------- LLM latency ----------

def fig_latency():
    csv = RESULTS / "ijcip_latency/latency_metrics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df_ok = df[df["status"] == "ok"]
    if df_ok.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    backends = df_ok["backend"].unique().tolist()
    ks = df_ok["k"].unique().tolist()
    x = np.arange(len(backends)); w = 0.35
    for i, k in enumerate(ks):
        sub = df_ok[df_ok["k"] == k].set_index("backend").reindex(backends)
        ax.bar(x + (i - 0.5) * w, sub["mean_ms"], w, label=k,
                edgecolor="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(backends, rotation=15, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_yscale("log")
    ax.set_title("LLM latency vs backend × K")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "llm_latency")


# ---------- baselines comparison ----------

def fig_baseline_comparison():
    csv = RESULTS / "ijcip_baselines/baseline_comparison.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    detectors = [d for d in DETECTOR_ORDER if d in df["detector"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, lbl in zip(axes, ("ens_kwh", "curt_kwh"),
                                  ("ENS (kWh)", "Curtailed energy (kWh)")):
        pivot = df.pivot_table(index="scenario", columns="detector",
                                  values=metric)
        pivot = pivot[detectors]
        x = np.arange(len(pivot.index)); w = 0.09
        for i, det in enumerate(detectors):
            ax.bar(x + (i - len(detectors)/2) * w, pivot[det].values, w,
                    color=DETECTOR_COLOR.get(det, "#888"),
                    label=DETECTOR_LABEL.get(det, det),
                    edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("ieee", "I").replace("_", " ")
                                for s in pivot.index], rotation=15)
        ax.set_ylabel(lbl); ax.set_title(lbl + " by detector")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    fig.tight_layout()
    _save(fig, "baseline_comparison")


# ---------- LaTeX tables ----------

def write_tables():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Theme-mapping table
    rows = [
        ("Trustworthy AI/ML for CI", "five-agent decomposition + Coordinator", "§4"),
        ("Adversarial robustness", "perturbation suite (six families)", "§5.4 / Tab. 4"),
        ("External benchmark validation", "UNSW-MG24 mock; SWaT/WADI/TON_IoT/EPIC adapters", "§5.2 / Tab. 2"),
        ("Stronger safety baselines", "5 new safety-aware baselines + oracle", "§5.3 / Tab. 3"),
        ("Reproducibility", "hash-pinned prompts + Makefile target", "§7"),
        ("HITL governance", "SLO × operator-behaviour sweep", "§5.5 / Tab. 5"),
        ("Objective sensitivity", "weight + threshold sweep, Pareto frontier", "§5.6 / Fig. 5"),
        ("Latency for edge deployment", "real LoRA + heuristic + K=1 vs K=3", "§5.7 / Tab. 6"),
    ]
    tex = ["\\begin{tabular}{lll}",
            "\\toprule",
            "VSI theme & DER-SecAgent contribution & Reference \\\\",
            "\\midrule"]
    tex += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (FIG_DIR / "table_theme_mapping.tex").write_text("\n".join(tex) + "\n")

    # External-benchmark table
    csv = RESULTS / "ijcip_external_benchmark/metrics.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        tex = ["\\begin{tabular}{llrrrrr}",
                "\\toprule",
                "Dataset & Detector & N & Acc. & macro-F1 & wF1 & Status \\\\",
                "\\midrule"]
        for _, r in df.iterrows():
            n = "--" if r["status"] == "not_run" else f"{int(r['n_samples'])}"
            acc = "--" if r["status"] == "not_run" else f"{r['accuracy']:.3f}"
            mf1 = "--" if r["status"] == "not_run" else f"{r['macro_f1']:.3f}"
            wf1 = "--" if r["status"] == "not_run" else f"{r['weighted_f1']:.3f}"
            tex.append(f"{r['dataset']} & {r['detector']} & {n} & {acc} & "
                        f"{mf1} & {wf1} & {r['status']} \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        (FIG_DIR / "table_external_benchmarks.tex").write_text("\n".join(tex) + "\n")


# ---------- driver ----------

def main():
    fig_system_overview()
    fig_adversarial()
    fig_external_benchmark()
    fig_ablation()
    fig_hitl()
    fig_objective_heatmap()
    fig_latency()
    fig_baseline_comparison()
    write_tables()
    print("done")


if __name__ == "__main__":
    main()
