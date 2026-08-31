"""Render every figure used in the paper from `results/`.

Single entry point: `python -m code.evaluation.build_figures`. Each figure
is small, self-contained, and saved both as PDF (for the paper) and PNG
(for slide-deck use). Headless matplotlib only --- no notebook dependency.

No monetary aggregation appears anywhere; every figure plots a raw
physical metric (ENS, curtailment, voltage band excursion, frequency
deviation) or a behavioural rate (FP-action, FN-action, latency).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = Path("paper/figure")
RESULTS = Path("code/results")

DETECTOR_ORDER = ["rule_ids", "prior_mas", "single_llm", "der_secagent"]
DETECTOR_LABEL = {
    "rule_ids":     "rule_ids",
    "prior_mas":    "prior_mas",
    "single_llm":   "single_llm",
    "der_secagent": "DER-SecAgent (ours)",
}
DETECTOR_COLOR = {
    "rule_ids":     "#888888",
    "prior_mas":    "#3b7dd8",
    "single_llm":   "#e07a3a",
    "der_secagent": "#2e7a3a",
}

SCENARIO_SHORT = {
    "ieee13_fdi_inverter":         "IEEE-13 FDI",
    "ieee13_command_spoof":        "IEEE-13 cmd-spoof",
    "ieee34_command_spoof_derms":  "IEEE-34 DERMS spoof",
}

METRIC_LABEL = {
    "ens_kwh":        "ENS (kWh)",
    "curt_kwh":       "Curtailed energy (kWh)",
    "voltage_frac":   "Voltage band excursion (fraction)",
    "freq_dev_hz":    "Frequency deviation (Hz)",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  wrote {FIG_DIR / name}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Fig: per-metric grouped bars across scenarios

def fig_metric_by_scenario(metric: str, fname: str) -> None:
    df = pd.read_csv(RESULTS / "physical_metrics.csv")
    sub = df[df["threshold"].between(0.49, 0.51)]
    agg = (sub.groupby(["scenario", "detector"], as_index=False)[metric].mean())
    pivot = agg.pivot(index="scenario", columns="detector", values=metric)
    pivot = pivot[DETECTOR_ORDER]
    scenarios = list(pivot.index)
    x = np.arange(len(scenarios))
    w = 0.18

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ymax = 0.0
    for i, det in enumerate(DETECTOR_ORDER):
        vals = pivot[det].values
        bars = ax.bar(x + (i - 1.5) * w, vals, w,
                      color=DETECTOR_COLOR[det], label=DETECTOR_LABEL[det],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01 * (max(vals) if max(vals) > 0 else 1),
                    f"{v:.3g}", ha="center", va="bottom", fontsize=7)
        ymax = max(ymax, float(np.nanmax(vals)) if len(vals) else 0.0)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax.set_ylabel(METRIC_LABEL[metric] + " (lower=better)")
    ax.set_title(f"{METRIC_LABEL[metric]} by scenario × detector "
                 f"(mean over 5 seeds)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.25)
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Fig: macro-F1 grouped bars (per-scenario)

def fig_macro_f1() -> None:
    df = pd.read_csv(RESULTS / "macro_f1_by_run.csv")
    agg = (df.groupby(["scenario", "detector"], as_index=False)["macro_f1"]
              .mean())
    pivot = agg.pivot(index="scenario", columns="detector", values="macro_f1")
    pivot = pivot[DETECTOR_ORDER]
    scenarios = list(pivot.index)
    x = np.arange(len(scenarios))
    w = 0.18

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, det in enumerate(DETECTOR_ORDER):
        vals = pivot[det].values
        ax.bar(x + (i - 1.5) * w, vals, w,
               color=DETECTOR_COLOR[det], label=DETECTOR_LABEL[det],
               edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_SHORT[s] for s in scenarios])
    ax.set_ylabel("Macro-F1 (excluding 'none' class)")
    ax.set_title("Detection macro-F1 by scenario × detector (mean over 5 seeds)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "macro_f1_by_scenario")


# ---------------------------------------------------------------------------
# Fig: caution-metrics 3-panel (FP, FN, unsafe rates)

def fig_caution_metrics() -> None:
    df = pd.read_csv(RESULTS / "caution_metrics.csv")
    metrics = [("fp_action_rate",   "FP-action rate (quiet steps)"),
               ("fn_action_rate",   "FN-action rate (attack steps)"),
               ("unsafe_action_rate", "Unsafe-command rate (severe tier)")]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=False)
    for ax, (col, title) in zip(axes, metrics):
        agg = (df.groupby("detector", as_index=False)[col].mean())
        agg = agg.set_index("detector").reindex(DETECTOR_ORDER)
        bars = ax.bar(range(len(agg)), agg[col].values,
                      color=[DETECTOR_COLOR[d] for d in agg.index],
                      edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, agg[col].values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels([DETECTOR_LABEL[d] for d in agg.index],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Caution Agent quantitative metrics (avg over scenarios × seeds)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "caution_metrics")


# ---------------------------------------------------------------------------
# Fig: latency boxplot + tail-latency bars

def fig_latency() -> None:
    df = pd.read_csv(RESULTS / "runtime.csv")
    df["mean_us"] = df["mean_ms"] * 1000.0
    df["p95_us"] = df["p95_ms"] * 1000.0
    df["p99_us"] = df["p99_ms"] * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    data = [df[df["detector"] == d]["mean_us"].values for d in DETECTOR_ORDER]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.55,
                         tick_labels=[DETECTOR_LABEL[d] for d in DETECTOR_ORDER])
    for patch, det in zip(bp["boxes"], DETECTOR_ORDER):
        patch.set_facecolor(DETECTOR_COLOR[det])
        patch.set_alpha(0.85)
    axes[0].set_ylabel("Mean step latency (µs)")
    axes[0].set_title("Per-step mean latency")
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.3)
    for label in axes[0].get_xticklabels():
        label.set_rotation(20); label.set_ha("right")

    x = np.arange(len(DETECTOR_ORDER))
    w = 0.4
    p95 = [df[df["detector"] == d]["p95_us"].mean() for d in DETECTOR_ORDER]
    p99 = [df[df["detector"] == d]["p99_us"].mean() for d in DETECTOR_ORDER]
    axes[1].bar(x - w / 2, p95, w, label="p95",
                color=[DETECTOR_COLOR[d] for d in DETECTOR_ORDER],
                edgecolor="black", linewidth=0.6, alpha=0.8)
    axes[1].bar(x + w / 2, p99, w, label="p99",
                color=[DETECTOR_COLOR[d] for d in DETECTOR_ORDER],
                edgecolor="black", linewidth=0.6, alpha=0.5, hatch="//")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([DETECTOR_LABEL[d] for d in DETECTOR_ORDER],
                            rotation=20, ha="right")
    axes[1].set_ylabel("Tail latency (µs)")
    axes[1].set_yscale("log")
    axes[1].set_title("p95 vs p99 tail latency")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Detector latency under heuristic-fallback configuration",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, "latency")


# ---------------------------------------------------------------------------
# Fig: paired-bootstrap forest plot on the headline ENS metric

def fig_bootstrap_forest() -> None:
    rows = json.loads((RESULTS / "pairwise_bootstrap.json").read_text())
    fig, ax = plt.subplots(figsize=(7.5, 2.6))

    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        pe = r["point_estimate"]
        lo = r["ci_low"]
        hi = r["ci_high"]
        color = "#2e7a3a" if r.get("reject") else "#888888"
        ax.errorbar([pe], [i], xerr=[[pe - lo], [hi - pe]],
                    fmt="o", color=color, capsize=4, elinewidth=2)
        # annotate with adjusted p-value, just to the right of the bar
        ax.text(hi + abs(hi - lo) * 0.04, i,
                f"p*={r.get('p_adj_holm', float('nan')):.3g}",
                va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['a']} − {r['b']}" for r in rows])
    ax.set_xlabel("Δ mean ENS (kWh); positive favors DER-SecAgent")
    ax.set_title("Paired-bootstrap forest plot vs DER-SecAgent (headline metric: ENS)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "bootstrap_forest")


# ---------------------------------------------------------------------------
# Fig: ENS-vs-threshold curves (replaces the old cost-vs-threshold curves)

def fig_threshold_curves() -> None:
    df = pd.read_csv(RESULTS / "physical_metrics.csv")
    metrics = [("ens_kwh", "ENS (kWh)"), ("curt_kwh", "Curtailed energy (kWh)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, (metric, label) in zip(axes, metrics):
        agg = (df.groupby(["detector", "threshold"], as_index=False)[metric].mean())
        for det in DETECTOR_ORDER:
            g = agg[agg["detector"] == det].sort_values("threshold")
            ax.plot(g["threshold"], g[metric], marker="o",
                    color=DETECTOR_COLOR[det], label=DETECTOR_LABEL[det],
                    linewidth=2)
        ax.set_xlabel("Detector confidence threshold")
        ax.set_ylabel(label)
        ax.set_title(label + " vs. threshold")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, "threshold_curves")


# ---------------------------------------------------------------------------
# Fig: case-study timeline

def fig_case_study() -> None:
    """Render the case-study figure as a 2-panel timeline."""
    import sys
    sys.path.insert(0, ".")
    import importlib
    DERSecAgent = importlib.import_module("code.Multi_AI_Agent.adapter").DERSecAgentDetector
    StubFeeder = importlib.import_module("code.simulation.feeder").StubFeeder
    run_scenario = importlib.import_module("code.simulation.harness").run_scenario

    cfg_path = "code/simulation/scenarios/ieee13_command_spoof/config.yaml"
    import yaml
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    feeder = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"])
    out = run_scenario(cfg_path, DERSecAgent(), seed=0,
                       out_root="code/results/runs_for_figures", feeder=feeder)
    ts = pd.read_csv(out / "timeseries.csv")

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    ax_p, ax_v = axes
    p_col = next(c for c in ts.columns if c.startswith("p_pv_INV_634"))
    v_col = next(c for c in ts.columns if c.startswith("v_pu_634"))

    ax_p.plot(ts["t"], ts[p_col], color="#2e7a3a", linewidth=1.8,
              label="INV_634 dispatch (kW)")
    ax_p.axvspan(180, 300, color="#e07a3a", alpha=0.15, label="attack window")
    ax_p.axvline(180, color="#e07a3a", linestyle="--", linewidth=0.8)
    ax_p.set_ylabel("Active power (kW)")
    ax_p.set_title("Case study: ieee13_command_spoof --- DER-SecAgent timeline")
    ax_p.legend(loc="lower right", fontsize=8)
    ax_p.grid(True, alpha=0.3)

    ax_v.plot(ts["t"], ts[v_col], color="#3b7dd8", linewidth=1.6,
              label="bus 634 voltage (pu)")
    ax_v.axvspan(180, 300, color="#e07a3a", alpha=0.15)
    ax_v.axhline(0.95, color="black", linestyle=":", linewidth=0.6,
                 label="ANSI C84.1 lower")
    ax_v.set_xlabel("time (s)")
    ax_v.set_ylabel("Voltage (pu)")
    ax_v.legend(loc="lower right", fontsize=8)
    ax_v.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "case_study_timeline")


# ---------------------------------------------------------------------------
# Fig: 5-agent architecture diagram

def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")

    nodes = [
        (0.6, "Telemetry\nAnalyst",  "deterministic\nFeatureView", "#dde6f7"),
        (2.4, "Hypothesis\nAgent",   "K=3 self-consistency\n(zero/CoT/ToT)", "#dde6f7"),
        (4.4, "Energy Impact\nAgent","per-action\n(E_curt, ENS, tier)",  "#dde6f7"),
        (6.4, "Caution\nAgent",      "4-rule\nsafety veto",           "#fbe5d6"),
        (8.4, "Coordinator\n(Critic)","argmin physical impact\n+ class-aware avoid", "#dde6f7"),
        (10.3,"Report\nAgent",       "hash-chained\naudit (post-hoc)", "#e2eada"),
    ]
    h = 1.4
    for x, name, sub, col in nodes:
        ax.add_patch(plt.Rectangle((x - 0.7, 1.1), 1.4, h, facecolor=col,
                                    edgecolor="black", linewidth=1.0))
        ax.text(x, 1.1 + h - 0.30, name, ha="center", va="center",
                fontsize=9, weight="bold")
        ax.text(x, 1.1 + 0.40, sub, ha="center", va="center",
                fontsize=7.5, color="#444")

    xs = [n[0] for n in nodes]
    for i in range(len(xs) - 1):
        x1 = xs[i] + 0.7
        x2 = xs[i + 1] - 0.7
        ax.annotate("", xy=(x2, 1.8), xytext=(x1, 1.8),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.text(0.6, 2.75, "telemetry &\nevent window",
            ha="center", va="center", fontsize=8, style="italic")
    ax.text(8.4, 2.75, "→ harness\n(action emission)",
            ha="center", va="center", fontsize=8, style="italic")
    ax.text(10.3, 2.75, "→ tamper-evident\nlog",
            ha="center", va="center", fontsize=8, style="italic")
    ax.text(5.5, 0.5,
            "single LangGraph state machine; field ownership in Tab. (Appendix B)",
            ha="center", va="center", fontsize=7.5, color="#666", style="italic")
    fig.tight_layout()
    _save(fig, "five_agent_architecture")


# ---------------------------------------------------------------------------
# Driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset; default: all")
    args = ap.parse_args()

    builders = {
        "architecture":  fig_architecture,
        "threshold":     fig_threshold_curves,
        "ens":           lambda: fig_metric_by_scenario("ens_kwh", "ens_by_scenario"),
        "curt":          lambda: fig_metric_by_scenario("curt_kwh", "curt_by_scenario"),
        "macro_f1":      fig_macro_f1,
        "caution":       fig_caution_metrics,
        "latency":       fig_latency,
        "forest":        fig_bootstrap_forest,
        "case_study":    fig_case_study,
    }
    keys = [k for k in args.only.split(",") if k.strip()] if args.only else list(builders.keys())
    for k in keys:
        fn = builders.get(k)
        if fn is None:
            continue
        print(f"[build_figures] {k}")
        fn()
    print("done")


if __name__ == "__main__":
    main()
