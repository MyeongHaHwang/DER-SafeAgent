#!/usr/bin/env python3
"""Export the runtime action policy to a single canonical view.

The executable source of truth for the action policy is
``code/Multi_AI_Agent/safety_projection.py`` (the constants the shield
actually runs). This script derives, deterministically, from those constants
plus the testbed action semantics in ``code/simulation``:

  * ``code/configs/action_policy.yaml``  — machine-readable policy export;
  * ``artifacts/tables/table_action_policy.tex`` — the manuscript's
    action-policy table (input after the shield subsection).

``tests/test_action_policy.py`` re-derives the export and fails if the YAML,
the manuscript table, the runtime constants, or the adversarial suite's
must-not sets drift apart. Run from the repository root:

    python3 scripts/export_action_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.Multi_AI_Agent.safety_projection import (   # noqa: E402
    CLASS_INAPPROPRIATE, CLASS_INEFFECTIVE, CONFIDENCE_FLOOR,
    IRREVERSIBLE, IRREVERSIBLE_CONFIDENCE, REGISTRY, deterministic_choice,
    safe_candidates)

SEVERITY_THRESHOLD = 0.30            # mirrors safe_candidates()
EVIDENCE_CLASSES = ["command_spoof", "replay", "dos", "fdi", "none"]
FALLBACK_ORDER = {                    # mirrors deterministic_choice()
    "command_spoof": ["freeze_setpoint", "throttle_ramp", "no_op"],
    "fdi": ["freeze_setpoint", "request_ied_revalidation", "no_op"],
    "replay": ["request_ied_revalidation", "freeze_setpoint", "no_op"],
    "dos": ["request_ied_revalidation", "freeze_setpoint", "no_op"],
    "firmware": ["request_ied_revalidation", "freeze_setpoint", "no_op"],
    "none": ["no_op"],
}

# Testbed semantics (code/simulation/feeder.py, StubFeeder + OpenDSS backend).
TESTBED_SEMANTICS = {
    "freeze_setpoint": ("hold last validated setpoint (testbed: pin to 70% of "
                        "rated kW); persists for the episode remainder"),
    "throttle_ramp": ("restrict ramping toward nominal (OpenDSS: InvControl "
                      "RiseFallLimit=1.667, the IEEE 1547 minimum)"),
    "isolate_inverter": ("disable the generator element (absorbing: output "
                         "stays removed until operator restoration)"),
    "request_ied_revalidation": "re-attestation request; no physical effect",
    "no_op": "monitor only",
}
RECOVERY = ("No automatic release is modelled: mitigations persist for the "
            "episode remainder; isolation is absorbing and requires operator "
            "restoration. Stand-down of the response requires the Evidence "
            "Gate's sustained-evidence condition to clear AND an operator or "
            "DERMS resume; model output alone can neither authorise an "
            "irreversible action (I3) nor stand the response down (I4).")
BASIS = ("Author-specified policy, motivated by the class mechanism (a "
         "telemetry-layer compromise must not be answered by disconnecting a "
         "healthy asset) and by counterfactual rollouts (throttle/revalidate "
         "are physically equivalent to no_op against a forced-zero DoS "
         "outage). The evaluation establishes conformance to this policy, "
         "not its independent operational validation.")


def class_partition(cls: str):
    """Partition the registry for one evidence class at nominal operating
    conditions (severity above threshold, confidence above the floor)."""
    allowed, removed = safe_candidates(cls, confidence=1.0,
                                       severity=max(SEVERITY_THRESHOLD, 0.31))
    autonomous = [a for a in allowed if a not in IRREVERSIBLE]
    # A model-proposed irreversible primitive is never auto-executed
    # (project(): always escalates); the deterministic fallback runs with
    # confidence 0, which keeps it out of reach there too.
    escalation_only = sorted(IRREVERSIBLE & set(allowed))
    if cls != "none":
        for a in IRREVERSIBLE:
            if a not in allowed and a not in escalation_only:
                pass  # removed as class-inappropriate; shown in `removed`
    return autonomous, escalation_only, removed


def build_yaml() -> str:
    lines = [
        "# Canonical action-policy export - GENERATED, do not edit by hand.",
        "# Source of truth: code/Multi_AI_Agent/safety_projection.py",
        "# Regenerate: python3 scripts/export_action_policy.py",
        "# Verified against the runtime constants by tests/test_action_policy.py",
        "",
        "registry: [%s]" % ", ".join(REGISTRY),
        "irreversible: [%s]" % ", ".join(sorted(IRREVERSIBLE)),
        f"severity_action_threshold: {SEVERITY_THRESHOLD}",
        f"confidence_floor: {CONFIDENCE_FLOOR}",
        f"irreversible_confidence_floor: {IRREVERSIBLE_CONFIDENCE}",
        "invariants:",
        "  I3_irreversible_escalation: model-proposed irreversible actions always escalate to a human; never auto-executed",
        "  I4_inaction_veto: model no_op cannot veto an incident established by deterministic protocol evidence",
        "  I5_deadline: a late or failed model call leaves the deterministic fast-path action in force",
        "classes:",
    ]
    for cls in EVIDENCE_CLASSES:
        autonomous, esc, removed = class_partition(cls)
        lines.append(f"  {cls}:")
        lines.append("    autonomous: [%s]" % ", ".join(autonomous))
        lines.append("    escalation_only: [%s]" % ", ".join(esc))
        lines.append("    removed:")
        for a in REGISTRY:
            if a in removed:
                lines.append(f"      {a}: {removed[a]}")
        lines.append("    deterministic_fallback_order: [%s]"
                     % ", ".join(FALLBACK_ORDER[cls]))
    lines.append("testbed_action_semantics:")
    for a in REGISTRY:
        lines.append(f"  {a}: {TESTBED_SEMANTICS[a]}")
    lines.append(f"recovery: {RECOVERY}")
    lines.append(f"basis: {BASIS}")
    return "\n".join(lines) + "\n"


_AB = {"no_op": "N", "freeze_setpoint": "F", "throttle_ramp": "T",
       "request_ied_revalidation": "R", "isolate_inverter": "I"}
_CLS = {"command_spoof": "Command spoof", "replay": "Replay", "dos": "DoS",
        "fdi": "FDI (rating residual)", "none": "No evidence / benign"}


def effective_fallback(cls: str) -> list[str]:
    allowed, _ = safe_candidates(cls, confidence=0.0,
                                 severity=max(SEVERITY_THRESHOLD, 0.31))
    return [a for a in FALLBACK_ORDER[cls] if a in allowed] or ["no_op"]


def build_tex() -> str:
    rows = []
    for cls in EVIDENCE_CLASSES:
        autonomous, esc, removed = class_partition(cls)
        auto = ", ".join(_AB[a] for a in autonomous)
        esc_s = ", ".join(_AB[a] for a in esc) if esc else "---"
        by_reason: dict[str, list[str]] = {}
        for a in REGISTRY:
            if a in removed:
                why = ("inappr." if "inappropriate" in removed[a] else
                       "ineff." if "ineffective" in removed[a] else
                       "conf." if "confidence" in removed[a] else "sev.")
                by_reason.setdefault(why, []).append(_AB[a])
        rem_s = "; ".join(f"{','.join(v)}\\,({k})"
                          for k, v in by_reason.items()) or "---"
        fb = r"$\rightarrow$".join(_AB[a] for a in effective_fallback(cls))
        rows.append(f"{_CLS[cls]} & {auto} & {esc_s} & {rem_s} & {fb} \\\\")
    body = "\n".join(rows)
    return (r"""\begin{table}[t]
\centering\small
\caption{Frozen class-aware action policy of the shield (source of truth:
\texttt{safety\_projection.py}; machine-readable export
\texttt{code/configs/action\_policy.yaml}, verified by
\texttt{tests/test\_action\_policy.py}). Actions: N = \textsc{no\_op},
F = \textsc{freeze\_setpoint}, T = \textsc{throttle\_ramp}, R =
\textsc{request\_ied\_revalidation}, I = \textsc{isolate\_inverter}
(irreversible). \emph{Autonomous} actions may execute without a human; a
model-proposed I is never auto-executed --- escalated to a human where the
class admits it (I3), vetoed with deterministic fallback where it does not
--- and the deterministic fallback never selects it. Non-N actions require
severity $\geq 0.30$; honouring a model choice requires stated confidence
$\geq 0.40$. Removals: inappr.\ = class-inappropriate, ineff.\ =
mechanism-ineffective. Testbed semantics: F holds the last validated
setpoint (70\% of rated kW) for the episode remainder; T restricts ramping
to the IEEE\,1547 minimum; I is absorbing and requires operator
restoration. No automatic release is modelled: stand-down requires the
gate's sustained evidence to clear and an operator/DERMS resume, never
model output alone (I4). The policy is author-specified, motivated by the
class mechanism and counterfactual rollouts; the evaluation establishes
conformance to it, not its independent operational validation.}
\label{tab:action-policy}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{lllll}
\toprule
Evidence class & Autonomous & Escalation-only & Removed & Fallback \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def main() -> int:
    (ROOT / "code/configs/action_policy.yaml").write_text(build_yaml())
    out_tex = ROOT / "artifacts/tables/table_action_policy.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(build_tex())
    print("[ok] code/configs/action_policy.yaml")
    print("[ok] artifacts/tables/table_action_policy.tex")
    # sanity: fallback order in this script must mirror deterministic_choice
    for cls, order in FALLBACK_ORDER.items():
        allowed = list(REGISTRY)
        assert deterministic_choice(cls, allowed) == order[0], cls
    return 0


if __name__ == "__main__":
    sys.exit(main())
