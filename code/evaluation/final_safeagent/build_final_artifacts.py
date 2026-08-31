"""Generate manuscript tables/figures + the final_numbers.tex macro file from
the tag's raw results. Nothing is hand-edited; re-running regenerates all.

Outputs:
  paper/tables/final/table_estimator.tex        (T6, P0-A)
  paper/tables/final/table_gate.tex             (T7, P0-B)
  paper/tables/final/table_opendss_llm.tex      (T8, P1-A)
  paper/tables/final/table_projection.tex       (T4, P1-B)
  paper/tables/final/table_ablation_final.tex   (T9a, P1-C)
  paper/tables/final/table_config_stats_final.tex (T9b, P1-D)
  paper/tables/final/final_numbers.tex          (macros for inline numbers)
  paper/figure/final/fig_estimator_repair.pdf
  paper/figure/final/fig_gate_tradeoff.pdf
  paper/figure/final/fig_opendss_llm.pdf
  paper/figure/final/fig_containment_final.pdf

Run: python3 -m code.evaluation.final_safeagent.build_final_artifacts
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAG = "ijcip_final_safeagent_20260810"
RES = Path("code/results") / TAG
TAB = Path("paper/tables/final")
FIG = Path("paper/figure/final")


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"[tex] {path}")


def t6_estimator() -> dict:
    rows = []
    nums = {}
    for split in ("dev", "holdout"):
        m = json.loads((RES / "p0a_estimator" / f"metrics_{split}.json").read_text())
        for est in ("E60", "EH"):
            d = m["estimators"][est]
            rows.append((est, split, d["top1_eps"], d["kendall_tau_mean"],
                         d["median_regret"], d["mean_regret"], d["p95_regret"],
                         d["mae_curtailment_kwh"]))
            nums[f"{est}{split}Top"] = d["top1_eps"]
            nums[f"{est}{split}MeanRegret"] = d["mean_regret"]
    body = "\n".join(
        f"{e} & {s} & {t:.2f} & {k:.2f} & {mr:.2f} & {meanr:.2f} & "
        f"{p95:.1f} & {mae:.1f} \\\\"
        for e, s, t, k, mr, meanr, p95, mae in rows)
    _w(TAB / "table_estimator.tex", r"""\begin{table}[t]
\centering\small
\caption{P0 Energy-Impact estimator repair: counterfactual validation of the
legacy 60\,s estimator (E60) and the horizon-aware estimator (EH) on the
development library (25 configurations) and the frozen causally-distinct
holdout (12 configurations). Top-1 counts an action optimal iff its realised
regret $\le 0.01$\,kWh-eq. EMH (multi-horizon marginal) is numerically
identical to EH on both splits and is omitted. The rollout oracle never
consults the estimator under evaluation.}
\label{tab:estimator}
\begin{tabular}{llcccccc}
\toprule
Estimator & Split & Top-1 & Kendall $\tau$ & Med.\ regret & Mean regret &
p95 regret & Curt.\ MAE \\
 & & & & (kWh-eq) & (kWh-eq) & (kWh-eq) & (kWh) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return nums


def t7_gate() -> dict:
    df = pd.read_csv(RES / "p0b_gate" / "p0b_summary.csv")
    nums = {}
    body = []
    label = {"D0-G0": "Deterministic, no gate (G0)",
             "D0-G1": "Deterministic + gate (G1)",
             "Q1-G0": "Qwen K=1, class-override, no gate (G0)",
             "Q1-G1": "Qwen K=1, class-override + gate (G1)",
             "QP-G0": "\\textbf{Final arch.} (proj.+veto), no gate (G0)",
             "QP-G1": "\\textbf{Final arch.} (proj.+veto) + gate (G1)"}
    for _, r in df.iterrows():
        if r.system not in label:
            continue
        body.append(f"{label[r.system]} & {r.benign_false_action_rate:.3f} & "
                    f"{r.benign_FA_normal:.2f}/{r.benign_FA_suspicious:.2f} & "
                    f"{r.attack_coverage:.3f} & {r.attack_ens_kwh_mean:.2f} & "
                    f"{int(r.benign_llm_calls)} & {int(r.hitl_total)} \\\\")
        nums[r.system.replace("-", "")] = dict(
            fa=r.benign_false_action_rate, cov=r.attack_coverage,
            ens=r.attack_ens_kwh_mean)
    _w(TAB / "table_gate.tex", r"""\begin{table}[t]
\centering\small
\caption{Incident Evidence Gate on the frozen 32-configuration test set
(16 benign incl.\ 8 near-boundary, 16 attacks incl.\ near-boundary cases;
thresholds frozen on the development split before any test run). Benign
false-action rate = fraction of benign configurations on which any
mitigation executed. FA normal/susp splits benign-normal from
benign-suspicious conditions; the residual G1 false actions are exactly the
three benign historian-echo configurations, where protocol-level evidence is
genuinely ambiguous. The final-architecture rows show the corrected
projection: the stand-down veto answers the echo cases with a
freeze-at-nominal (0\,kWh cost), and without the gate its escalation rule
floods the operator (614 HITL escalations from benign transients; 0 with
the gate).}
\label{tab:gate}
\begin{tabular}{lcccccc}
\toprule
System & Benign FA & FA norm/susp & Attack cov. & Attack ENS & Benign LLM
& HITL \\
 & rate & & & (kWh) & calls & \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return nums


def t8_opendss() -> dict:
    df = pd.read_csv(RES / "p1a_opendss" / "p1a_raw.csv")
    nums = {"opendssNonconv": int(df.n_nonconverged.sum())}
    attack = df[~df.is_control]
    g = (attack.groupby("system")
         .agg(curt=("curt_kwh", "mean"), ens=("ens_kwh", "mean"),
              vmin=("voltage_min_pu", "min"),
              vviol=("voltage_violation_frac", "mean"),
              lat=("mitigation_latency_s", "mean"),
              shield=("shield_intervened", "sum"),
              hitl=("n_hitl", "sum"), nllm=("n_llm_calls", "sum"))
         .reindex(["OD0", "OQ1", "OL1", "OQP", "OLP"]).dropna(how="all"))
    body = []
    for s, r in g.iterrows():
        body.append(f"{s} & {r.curt:.1f} & {r.ens:.2f} & {r.vmin:.3f} & "
                    f"{r.vviol:.3f} & "
                    f"{'--' if np.isnan(r.lat) else f'{r.lat:.0f}'} & "
                    f"{int(r.shield)} & {int(r.hitl)} \\\\")
        nums[f"od_{s}_curt"] = float(r.curt)
    _w(TAB / "table_opendss_llm.tex", r"""\begin{table}[t]
\centering\small
\caption{Real QLoRA $\rightarrow$ real OpenDSS power flow on the
pre-registered 12-configuration subset (11 attack + 1 benign control;
frozen before any model run). Every row aggregates real solves (zero
non-converged ticks); LLM rows abort on any serving fallback and record the
adapter SHA per call. ENS and the voltage-violation fraction are zero on
every row of this subset (healthy feeders) and are reported
descriptively; curtailment and $V_{\min}$ carry the evidence.}
\label{tab:opendss-llm}
\begin{tabular}{lccccccc}
\toprule
System & Curt. & ENS & $V_{\min}$ & V-viol & Mit.\ lat. & Shield & HITL \\
 & (kWh) & (kWh) & (pu) & frac & (s) & interv. & \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return nums


def t4_projection() -> dict:
    df = pd.read_csv(RES / "p1b_projection" / "p1b_raw.csv")
    nums = {}
    per = (df.groupby(["arm", "scenario"])[["ens_kwh", "curt_kwh"]]
           .mean().reset_index())
    g = (per.groupby("arm")
         .agg(ens=("ens_kwh", "mean"), curt=("curt_kwh", "mean")))
    ex = (df.groupby("arm")
          .agg(nllm=("n_llm_calls", "sum"), hon=("n_model_honoured", "sum"),
               fb=("n_det_fallback", "sum"), ov=("n_safety_override", "sum"),
               hitl=("n_hitl", "sum")))
    order = ["D0", "Q1", "L1", "QPROJ", "LPROJ", "OPROJ"]
    body = []
    for a in order:
        if a not in g.index:
            continue
        r, e = g.loc[a], ex.loc[a]
        body.append(f"{a} & {r.ens:.2f} & {r.curt:.2f} & {int(e.nllm)} & "
                    f"{int(e.hon)} & {int(e.fb)} & {int(e.ov)} & "
                    f"{int(e.hitl)} \\\\")
        nums[f"p1b_{a}_ens"] = float(r.ens)
    _w(TAB / "table_projection.tex", r"""\begin{table}[t]
\centering\small
\caption{Final-architecture cyber-physical outcomes over the frozen
25-configuration library $\times$ 3 genuine stochastic episodes (75 runs per
arm; configuration is the inference unit and episode means are aggregated
per configuration). All arms run behind the frozen Evidence Gate. Q1/L1 use
the class-override shield; QPROJ/LPROJ the corrected safety projection
(model-proposed irreversible primitives always escalate; EH ranks the safe
set on fallback); OPROJ feeds the ground-truth class through the same
projection logic. Proposal-honoured / fallback / override columns count
decision-steps --- the deterministic layers re-evaluate every triggered
tick --- not incidents.}
\label{tab:projection}
\begin{tabular}{lccccccc}
\toprule
Arm & ENS & Curt. & LLM & Proposal & Det. & Safety & HITL \\
 & (kWh) & (kWh) & calls & honoured & fallback & override & \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return nums


def t9_ablation() -> dict:
    df = pd.read_csv(RES / "p1c_ablation" / "p1c_by_arm_qwen.csv")
    nums = {}
    order = ["final_full", "minus_evidence_gate", "minus_projection",
             "minus_class_avoidance", "minus_caution_hitl",
             "minus_eh_fallback", "deterministic_only", "bare_llm"]
    label = {"final_full": "\\textbf{Final DER-SafeAgent}",
             "minus_evidence_gate": "$-$ Evidence Gate",
             "minus_projection": "$-$ projection (class-override)",
             "minus_class_avoidance": "$-$ class-aware avoidance",
             "minus_caution_hitl": "$-$ Caution/HITL",
             "minus_eh_fallback": "$-$ EH fallback (class table)",
             "deterministic_only": "deterministic-only",
             "bare_llm": "bare LLM (registry only)"}
    body = []
    for a in order:
        r = df[df.arm == a]
        if r.empty:
            continue
        r = r.iloc[0]
        body.append(f"{label[a]} & {int(r.proposed_irrev)} & "
                    f"{int(r.executed_irrev)} & {r.policy_violation:.3f} & "
                    f"{r.intervened:.2f} & {r.hitl_rate:.2f} & "
                    f"{r.correct_refusal:.2f} \\\\")
        nums[f"abl_{a}_viol"] = float(r.policy_violation)
        nums[f"abl_{a}_execirrev"] = int(r.executed_irrev)
    _w(TAB / "table_ablation_final.tex", r"""\begin{table}[t]
\centering\small
\caption{Final containment ablation: 42 adversarial cases (14 families,
real Qwen QLoRA, strict serving), each held as sustained input for 8
consecutive ticks so persistence-gated components can fire (the prior
single-step protocol could not exercise the gate). Irrev.\ = the
irreversible \textsc{isolate\_inverter} primitive.}
\label{tab:ablation-final}
\begin{tabular}{lcccccc}
\toprule
Configuration & Irrev. & Irrev. & Policy & Shield & HITL & Correct \\
 & proposed & \textbf{executed} & violation & interv. & rate & refusal \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    return nums


def t9_stats() -> None:
    df = pd.read_csv(RES / "statistics" / "final_stats.csv")
    keep = df[df.family.isin(["F1", "F3"])
              & ~df.metric.str.contains("descriptive")]
    body = []
    for _, r in keep.iterrows():
        ci = (f"[{r.ci95_lo:+.2f}, {r.ci95_hi:+.2f}]"
              if not np.isnan(r.ci95_lo) else "--")
        p = "--" if np.isnan(r.p_holm) else (f"{r.p_holm:.3f}"
                                             if r.p_holm >= 0.001 else "$<$0.001")
        body.append(f"{r.contrast} & {r.metric.replace('_', ' ')} & "
                    f"{int(r.n_configurations)} & {r.mean_diff:+.3f} & {ci} & "
                    f"{r.d_z:+.2f} & {p} \\\\")
    _w(TAB / "table_config_stats_final.tex", r"""\begin{table}[t]
\centering\small
\caption{Configuration-level statistics for the final architecture. Paired
two-sided sign-flip permutation tests on per-configuration differences
(episode means within configuration), paired bootstrap 95\% CIs, Cohen's
$d_z$, Holm correction within each planned family. F1: physical outcome vs
the deterministic fast path (P1-B). F3: benign false-action, gate vs no
gate (P0-B).}
\label{tab:stats-final}
\begin{tabular}{llccccc}
\toprule
Contrast & Metric & $N$ cfg & Mean diff & 95\% CI & $d_z$ & $p_{\text{Holm}}$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def figures() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # F: estimator repair
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    for ax, split in zip(axes, ("dev", "holdout")):
        sel = pd.read_csv(RES / "p0a_estimator" / f"selections_{split}.csv")
        for i, est in enumerate(("E60", "EH")):
            r = sel[sel.estimator == est].regret.to_numpy()
            ax.scatter(np.full_like(r, i) + np.random.default_rng(1)
                       .uniform(-0.12, 0.12, len(r)), r, s=18, alpha=0.7,
                       color=["#c44e52", "#4c72b0"][i])
        ax.set_xticks([0, 1], ["E60\n(legacy 60 s)", "EH\n(horizon-aware)"])
        ax.set_ylabel("realised regret (kWh-eq)" if split == "dev" else "")
        ax.set_title(f"{split} ({sel.scenario.nunique()} configs)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_estimator_repair.pdf")
    plt.close(fig)

    # F: gate trade-off
    df = pd.read_csv(RES / "p0b_gate" / "p0b_summary.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for _, r in df.iterrows():
        ax.scatter(r.benign_false_action_rate, r.attack_coverage, s=70,
                   color=("#4c72b0" if r.system.endswith("G1") else "#c44e52"))
        ax.annotate(r.system, (r.benign_false_action_rate, r.attack_coverage),
                    textcoords="offset points", xytext=(6, -4), fontsize=8)
    ax.set_xlabel("benign false-action rate (16 benign configs)")
    ax.set_ylabel("attack coverage (16 attacks)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gate_tradeoff.pdf")
    plt.close(fig)

    # F: OpenDSS curtailment by system
    try:
        df = pd.read_csv(RES / "p1a_opendss" / "p1a_raw.csv")
        attack = df[~df.is_control]
        piv = attack.pivot_table(index="configuration_id", columns="system",
                                 values="curt_kwh")
        cols = [c for c in ("OD0", "OQ1", "OL1", "OQP", "OLP") if c in piv]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        x = np.arange(len(piv.index))
        w = 0.8 / max(len(cols), 1)
        for i, c in enumerate(cols):
            ax.bar(x + i * w, piv[c], width=w, label=c)
        ax.set_xticks(x + 0.4 - w / 2, piv.index, rotation=45, ha="right",
                      fontsize=7)
        ax.set_ylabel("curtailed energy (kWh)")
        ax.legend(fontsize=8, ncol=len(cols))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "fig_opendss_llm.pdf")
        plt.close(fig)
    except FileNotFoundError:
        pass

    # F: containment (proposed vs executed irreversible, final ablation)
    try:
        df = pd.read_csv(RES / "p1c_ablation" / "p1c_by_arm_qwen.csv")
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        x = np.arange(len(df))
        ax.bar(x - 0.2, df.proposed_irrev, width=0.4,
               label="irreversible proposed", color="#dd8452")
        ax.bar(x + 0.2, df.executed_irrev, width=0.4,
               label="irreversible executed", color="#c44e52")
        ax.set_xticks(x, df.arm, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("count / 42 adversarial cases")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "fig_containment_final.pdf")
        plt.close(fig)
    except FileNotFoundError:
        pass
    print(f"[fig] -> {FIG}")


def main() -> None:
    nums = {}
    for fn in (t6_estimator, t7_gate, t8_opendss, t4_projection, t9_ablation):
        try:
            nums.update(fn() or {})
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")
    try:
        t9_stats()
    except FileNotFoundError as e:
        print(f"skip stats: {e}")
    figures()
    # numbers macro file
    lines = ["% auto-generated by build_final_artifacts.py - do not edit"]
    digit = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
             "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    for k, v in sorted(nums.items()):
        if isinstance(v, dict):
            continue
        name = "".join(digit.get(c, c) for c in k.title().replace("_", "")
                       if c.isalnum())
        val = f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
        lines.append(f"\\newcommand{{\\fn{name}}}{{{val}}}")
    _w(TAB / "final_numbers.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
