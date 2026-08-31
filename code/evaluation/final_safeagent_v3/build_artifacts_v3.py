"""V3 artifact builder — regenerate v3 tables/figures from canonical CSVs.

Produces the tables/figures for which canonical data exist; skips (and reports)
those whose inputs are pending. Nothing is hand-edited.

Run: python3 -m code.evaluation.final_safeagent_v3.build_artifacts_v3
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("code/results/ijcip_final_v3")
TAB = Path("artifacts/tables")
FIG = Path("artifacts/figures")
PROC = Path("artifacts/processed")

# Transparent, semantics-based display labels for the holdout arms (a labelling
# change only; the underlying CSV arm codes and numbers are unchanged).
#   bare = ungated, fixed-cadence model-only (diagnostic, NOT a matched ablation)
#   Q1/L1 = class-override shield ; QPROJ/LPROJ = safe projection ; OPROJ = oracle
DISPLAY = {
    "D0": "D0", "bareQ": "ModelOnly-Q", "bareL": "ModelOnly-L",
    "Q1": "ClassOnly-Q", "L1": "ClassOnly-L", "QPROJ": "SafeProj-Q",
    "LPROJ": "SafeProj-L", "OPROJ": "OracleClass-Proj",
}


def disp(arm):
    return DISPLAY.get(arm, arm)


def _w(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s)
    print(f"[tex] {p}")


def table_gate():
    p = RES / "gate_robustness" / "gate_v3_summary.csv"
    if not p.exists():
        print("[skip] gate table (pending)")
        return
    df = pd.read_csv(p)
    body = []
    label = {"benign_false_open": "Benign false-open",
             "gateopen_command_spoof": "Gate-open command-spoof",
             "gateopen_replay": "Gate-open replay",
             "gateopen_dos": "Gate-open DoS",
             "gateopen_fdi_high": "Gate-open FDI (high)",
             "gateopen_fdi_med": "Gate-open FDI (med)",
             "gateopen_fdi_low": "Gate-open FDI (low)"}
    for _, r in df.iterrows():
        if r["group"] not in label:
            continue
        body.append(f"{label[r['group']]} & {int(r['k'])}/{int(r['n'])} & "
                    f"{r['rate']:.2f} & [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] \\\\")
    _w(TAB / "table_gate_v3.tex", r"""\begin{table}[t]
\centering\small
\caption{Evidence Gate detector-side coverage: per-group gate-open and benign
false-open on the hash-frozen 32-configuration test set plus per-magnitude FDI
probes, using only observable evidence. Thresholds calibrated on development
data and frozen before this evaluation. Two-sided Clopper--Pearson 95\%
intervals. Sub-capacity FDI (medium/low) has no observable hard signature and
is reported as a coverage miss.}
\label{tab:gate-v3}
\begin{tabular}{lccc}
\toprule
Group & $k/n$ & Rate & 95\% CI \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_eh():
    p = RES / "eh_robustness" / "eh_robustness.csv"
    if not p.exists():
        print("[skip] EH table (pending)")
        return
    df = pd.read_csv(p)
    eh = df[df.estimator == "EH"]
    e60 = df[df.estimator == "E60"]
    body = []
    # Display labels for the robustness conditions (labelling only; the CSV
    # condition codes and all numbers are unchanged).
    COND = {"nominal": "Nominal", "horizon_-50%": r"Horizon $-50\%$",
            "horizon_-25%": r"Horizon $-25\%$", "horizon_+25%": r"Horizon $+25\%$",
            "horizon_+50%": r"Horizon $+50\%$", "delayed_10s": r"Delayed 10\,s",
            "delayed_30s": r"Delayed 30\,s", "partial_failure": "Partial failure",
            "w_ens=3": r"$w_{\text{ENS}}=3$", "w_ens=8": r"$w_{\text{ENS}}=8$"}
    def _esc(s):  # escape LaTeX specials in unmapped condition labels
        return str(s).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%')
    for cond in df.condition.unique():
        a = eh[eh.condition == cond].iloc[0]
        b = e60[e60.condition == cond].iloc[0]
        lbl = COND.get(cond, _esc(cond)).ljust(18)
        body.append(f"{lbl} & {a['top1']:.2f} & "
                    f"{a['median_regret']:.2f} & {a['worst_regret']:.1f} & "
                    f"{b['top1']:.2f} \\\\")
    _w(TAB / "table_eh_v3.tex", r"""\begin{table}[t]
\centering\small
\caption{Horizon-aware estimator (EH) robustness: tie-aware top-1 optimal-action
rate and regret under horizon misspecification, delayed/partial execution, and
alternative cost weights, on the 12-configuration frozen estimator holdout. EH's
ranking is stable because the five-action choice is structurally determined;
point magnitudes remain imprecise (MAE $\approx$26\,kWh), so no
magnitude-prediction claim is made.}
\label{tab:eh-v3}
\begin{tabular}{lcccc}
\toprule
Condition & EH top-1 & EH med.\ regret & EH worst & E60 top-1 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_holdout():
    p = RES / "holdout_e2e" / "holdout_e2e_raw.csv"
    if not p.exists():
        print("[skip] holdout table (pending)")
        return
    df = pd.read_csv(p)
    per = (df.groupby(["arm", "scenario", "kind"])
           [["ens_kwh", "curt_kwh", "n_actions", "gate_open_covered",
             "n_llm_calls", "n_hitl"]].mean().reset_index())
    atk = per[per.kind == "attack"]
    ben = per[per.kind == "benign"]
    order = [a for a in ["D0", "bareQ", "bareL", "Q1", "L1", "QPROJ", "LPROJ",
                         "OPROJ"] if a in per.arm.unique()]
    # Ungated bare arms have no Evidence Gate, so gate-open coverage does not
    # apply and is reported N/A (never a misleading 0.000).
    UNGATED = {"bareQ", "bareL"}
    body = []
    for arm in order:
        a = atk[atk.arm == arm]
        b = ben[ben.arm == arm]
        gate = "N/A" if arm in UNGATED else f"{a.gate_open_covered.mean():.3f}"
        body.append(f"{disp(arm)} & {a.ens_kwh.mean():.3f} & {a.curt_kwh.mean():.2f} & "
                    f"{gate} & "
                    f"{(b.n_actions > 0).mean():.3f} & "
                    f"{int(a.n_llm_calls.sum())} & {int(a.n_hitl.sum())} \\\\")
    _w(TAB / "table_holdout_v3.tex", r"""\begin{table}[t]
\centering\small
\caption{Held-out end-to-end evaluation: 49 configurations (33 attack, 16
benign) $\times$ 3 stochastic episodes, observable-only runtime.
Configuration is the inference unit; episodes averaged within configuration.
\emph{Closed-loop coverage} is whether the incident is flagged at any tick of
the attack window in the closed loop; it is arm-dependent because each arm's
action feeds back into telemetry and the sustained-evidence gate (e.g.\ time to
first gate-open is $20$\,s for D0 vs.\ $111$\,s for L1), and is \emph{N/A} for
the ungated bare arms. It is \emph{not} the deterministic pre-action gate, which
is model-independent and identical across arms for identical input (verified:
$11{,}020$ pre-action ticks, $0$ violations); the detector-side gate recall is
Table~\ref{tab:gate-v3}. \emph{LLM calls} = sum over the 33 attack configs of
per-config mean calls (a sustained incident re-consults across ticks). Benign
false-action (FA) is physical-side. LLM arms run under strict serving (zero
fallbacks). Paired statistics in Table~\ref{tab:stats-primary}.}
\label{tab:holdout-v3}
\begin{tabular}{lcccccc}
\toprule
Arm & ENS & Curt. & Closed-loop & Benign & LLM & HITL \\
 & (kWh) & (kWh) & coverage & FA rate & calls & \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    # figure: per-config ENS diff vs D0
    d0 = atk[atk.arm == "D0"].set_index("scenario").ens_kwh
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    xs = [a for a in order if a != "D0"]
    for i, arm in enumerate(xs):
        a = atk[atk.arm == arm].set_index("scenario").ens_kwh
        common = a.index.intersection(d0.index)
        diff = (a.loc[common] - d0.loc[common]).to_numpy()
        ax.scatter(np.full(len(diff), i) + RNG_JIT(len(diff)), diff, s=14,
                   alpha=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(xs)), [disp(a) for a in xs], rotation=30,
                  ha="right", fontsize=8)
    ax.set_ylabel("ENS diff vs D0 (kWh), per config")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_holdout_ens_diff.pdf")
    fig.savefig(FIG / "fig_holdout_ens_diff.png", dpi=300)
    plt.close(fig)
    print(f"[fig] {FIG/'fig_holdout_ens_diff.pdf'}")


def adversarial_consultation():
    """Honest diagnostic: on how many cases was the model actually consulted
    (proposed a non-no_op action), i.e. the gate opened under sustained input.
    Replaces the retracted family-based 'regime' split (see
    ADVERSARIAL_REGIME_NOTE.md)."""
    for be in ("qwen", "llama"):
        p = RES / "adversarial" / f"adversarial_raw_{be}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["consulted"] = df.proposed != "no_op"
        g = (df.groupby("arm")
             .agg(n=("case_index", "count"),
                  consulted=("consulted", "sum"),
                  irrev_exec=("executed_irrev", "sum"),
                  policy_viol=("policy_violation", "sum"))
             .reset_index())
        PROC.mkdir(parents=True, exist_ok=True)
        g.to_csv(PROC / f"adversarial_consultation_{be}.csv", index=False)
        print(f"[csv] adversarial consultation {be}")


def table_adversarial():
    adversarial_consultation()
    for be in ("qwen", "llama"):
        p = RES / "adversarial" / f"adversarial_summary_{be}.csv"
        if not p.exists():
            print(f"[skip] adversarial table {be} (pending)")
            continue
        df = pd.read_csv(p)
        # Display labels for the ablation arms (labelling only; the CSV arm
        # codes and all numbers are unchanged).
        ADV = {"final": "Full arch.", "minus_evidence_gate": "$-$Evidence gate",
               "minus_projection": "$-$Projection",
               "minus_class_avoidance": "$-$Class avoidance",
               "minus_caution_hitl": "$-$Caution/HITL",
               "minus_eh_fallback": "$-$EH fallback",
               "deterministic_only": "Deterministic only",
               "bare_llm": "Unshielded"}
        body = []
        for _, r in df.iterrows():
            n = int(r['n'])
            ip = int(r['irrev_proposed'])
            ie = int(r['irrev_executed'])
            fv = int(round(r['policy_violation_rate'] * n))
            body.append(f"{ADV.get(r['arm'], r['arm'].replace('_', '\\_'))} & {n} & "
                        f"{ip}/{n} & {ie}/{n} & {r['irrev_exec_upper95']:.3f} & "
                        f"{fv}/{n} \\\\")
        _w(TAB / f"table_adversarial_v3_{be}.tex", (
            r"\begin{table}[t]\centering\small" "\n"
            r"\caption{Canonical 70-case adversarial containment (" + be +
            r"), sustained input, observable-only runtime. Denominator = "
            r"adversarial case ($n{=}70$). \emph{Irrev.\ prop.}\ and "
            r"\emph{Irrev.\ exec.}\ are the irreversible-proposal and "
            r"irreversible-\emph{execution} counts (action $=$ "
            r"\texttt{isolate\_inverter}); \emph{Forbidden act.}\ is the "
            r"distinct forbidden-action count (executed action in the case's "
            r"must-not set). Zero-execution arms carry a two-sided "
            r"Clopper--Pearson 95\% upper bound (\emph{exec.\ up95}). The "
            r"correct-refusal rate and full per-arm metrics are in the "
            r"supplementary material.}"
            "\n\\label{tab:adv-v3-" + be + "}\n"
            r"\resizebox{\columnwidth}{!}{%" "\n"
            r"\begin{tabular}{lccccc}" "\n\\toprule\n"
            r"Arm & $n$ & Irrev.\ prop. & Irrev.\ exec. & exec.\ up95 & "
            r"Forbidden act. \\" "\n\\midrule\n"
            + "\n".join(body) +
            "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"))


def table_stats_primary():
    """Primary paired-comparison statistics table (holdout ENS vs D0)."""
    p = RES / "statistics" / "F1_physical.csv"
    if not p.exists():
        print("[skip] primary stats table (pending)")
        return
    df = pd.read_csv(p)
    df = df[df.metric == "ens_kwh"].copy()
    order = ["bareQ-D0", "bareL-D0", "Q1-D0", "L1-D0", "QPROJ-D0",
             "LPROJ-D0", "OPROJ-D0"]
    df["o"] = df.contrast.map({c: i for i, c in enumerate(order)})
    df = df.sort_values("o")
    body = []
    for _, r in df.iterrows():
        praw = "$<$0.001" if r.p_raw < 0.001 else f"{r.p_raw:.3f}"
        pholm = "$<$0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
        sig = "\\,*" if r.p_holm < 0.05 else ""
        lbl = disp(r.contrast[:-3]) + r"$-$D0"   # every contrast is "<arm>-D0"
        body.append(f"{lbl} & {r.mean_diff:+.3f} & "
                    f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] & {r.d_z:+.2f} & "
                    f"{int(r.n_cfg)} & {praw} & {pholm}{sig} \\\\")
    _w(TAB / "table_stats_primary.tex", r"""\begin{table}[t]
\centering\small
\caption{Primary paired configuration-level comparison on the held-out
49-configuration set: attack-window ENS difference of each arm minus the
deterministic fast path (D0), episodes averaged within configuration. Paired
bootstrap 95\% CI, Cohen's $d_z$, paired sign-flip permutation $p$, and
Holm-adjusted $p$ across this primary family; resampling draws are shared
across contrasts (common random numbers), so identical difference vectors
receive identical intervals (ModelOnly-Q/L). ``*'' marks Holm $p<0.05$. The
bootstrap CI can exclude zero while the adjusted test does not (OracleClass-Proj),
which we report as a modest, not-significant effect. ModelOnly arms are
diagnostic, not matched shield ablations (\S\ref{sec:rq3}).}
\label{tab:stats-primary}
\begin{tabular}{lcccccc}
\toprule
Contrast (ENS) & Mean diff & 95\% CI & $d_z$ & $N$ & $p_{\text{raw}}$ & $p_{\text{Holm}}$ \\
 & (kWh) & (kWh) & & cfg & & \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def fig_containment_v3():
    p = RES / "adversarial" / "adversarial_summary_qwen.csv"
    if not p.exists():
        return
    order = ["bare_llm", "minus_class_avoidance", "minus_projection",
             "minus_evidence_gate", "minus_caution_hitl", "minus_eh_fallback",
             "deterministic_only", "final"]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    q = pd.read_csv(p).set_index("arm")
    x = np.arange(len(order))
    prop = [int(q.loc[a, "irrev_proposed"]) for a in order]
    exe = [int(q.loc[a, "irrev_executed"]) for a in order]
    ax.bar(x - 0.2, prop, width=0.4, label="irreversible proposed",
           color="#dd8452")
    ax.bar(x + 0.2, exe, width=0.4, label="irreversible executed",
           color="#c44e52")
    ax.set_xticks(x, [a.replace("_", "\n") for a in order], fontsize=7)
    ax.set_ylabel("count / 70 cases (Qwen)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_containment_v3.pdf")
    fig.savefig(FIG / "fig_containment_v3.png", dpi=300)
    plt.close(fig)
    print(f"[fig] {FIG/'fig_containment_v3.pdf'}")


def fig_gate_v3():
    p = RES / "gate_robustness" / "gate_v3_summary.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    rows = [("command-spoof", "gateopen_command_spoof"),
            ("replay", "gateopen_replay"), ("DoS", "gateopen_dos"),
            ("FDI high", "gateopen_fdi_high"), ("FDI med", "gateopen_fdi_med"),
            ("FDI low", "gateopen_fdi_low"), ("benign\n(false-open)",
                                             "benign_false_open")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    labels, rates, los, his, colors = [], [], [], [], []
    for lab, g in rows:
        r = df[df.group == g]
        if r.empty:
            continue
        r = r.iloc[0]
        labels.append(lab)
        rates.append(r["rate"])
        los.append(r["rate"] - r["ci_lo"])
        his.append(r["ci_hi"] - r["rate"])
        colors.append("#c44e52" if g == "benign_false_open" else "#4c72b0")
    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, yerr=[los, his], capsize=3)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("gate-open rate (95% CI)")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gate_v3.pdf")
    fig.savefig(FIG / "fig_gate_v3.png", dpi=300)
    plt.close(fig)
    print(f"[fig] {FIG/'fig_gate_v3.pdf'}")


def table_latency():
    p = Path("code/results/ijcip_final/latency/latency_distributions.csv")
    if not p.exists():
        print("[skip] latency table (pending)")
        return
    d = pd.read_csv(p)
    BE = {"qwen": "Qwen", "llama": "Llama"}
    body = []
    for _, r in d.iterrows():
        body.append(f"{BE.get(r['backend'], r['backend'])} & $K{{=}}{int(r['k'])}$ & {int(r['n_calls'])} & "
                    f"{r['mean_s']:.2f} & {r['median_s']:.2f} & {r['sd_s']:.2f} & "
                    f"{r['p95_s']:.2f} & {r['p99_s']:.2f} & {r['max_s']:.2f} \\\\")
    _w(TAB / "table_latency_v3.tex", r"""\begin{table}[t]
\centering\small
\caption{Per-backbone, per-$K$ consultation-latency distribution, from the
146-call ($K{=}1$) / 48-call ($K{=}3$) measurements; single-call ($K{=}1$)
inference is $\approx$5.5\,s on both backbones. All latency values in the paper
are regenerated from this measurement file. Adapter load is a one-off startup
cost, not part of per-call latency.}
\label{tab:latency-v3}
\begin{tabular}{llccccccc}
\toprule
Backend & $K$ & $n$ & mean & median & s.d. & p95 & p99 & max \\
 & & & (s) & (s) & (s) & (s) & (s) & (s) \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def RNG_JIT(n):
    return np.random.default_rng(1).uniform(-0.12, 0.12, n)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    table_gate()
    table_eh()
    table_holdout()
    fig_containment_v3()
    fig_gate_v3()
    table_latency()
    table_adversarial()
    table_stats_primary()
    print("v3 artifacts built (pending inputs skipped and reported)")


if __name__ == "__main__":
    main()
