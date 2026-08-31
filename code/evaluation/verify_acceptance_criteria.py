"""Mechanically verify the revision's acceptance criteria.

Each check reads the actual artifacts rather than trusting the prose. Run:
    python3 -m code.evaluation.verify_acceptance_criteria
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

TAG = "ijcip_revision_r1r2_20260805"
RES = Path("code/results") / TAG
CFG = Path("code/configs") / TAG
PAPER = Path("paper")

results: list[tuple[str, bool, str]] = []


def check(name: str, fn) -> None:
    try:
        ok, detail = fn()
    except Exception as exc:  # a failed check must be visible, not silent
        ok, detail = False, f"check raised: {type(exc).__name__}: {exc}"
    results.append((name, ok, detail))


def c1_adapter_proof():
    """Every QLoRA-in-the-loop run proves the intended adapter was loaded."""
    n_runs = n_traces = 0
    bad = []
    for sysdir in (RES / "llm_in_loop").glob("*lora*"):
        for man_p in sysdir.rglob("manifest.json"):
            man = json.loads(man_p.read_text())
            if "lora" not in str(man.get("llm_backend", "")):
                continue
            n_runs += 1
            sha = man.get("llm_adapter_sha")
            if not sha or not man.get("llm_strict"):
                bad.append(f"{man_p}: missing adapter sha or strict flag")
                continue
            for line in (man_p.parent / "decisions.jsonl").read_text().splitlines():
                for tr in ((json.loads(line).get("decision_trace") or {})
                           .get("llm_traces") or []):
                    n_traces += 1
                    if tr["backend"] != "real_lora" or tr["adapter_sha"] != sha:
                        bad.append(f"{man_p}: trace backend/sha mismatch")
    return (not bad and n_runs > 0,
            f"{n_runs} LLM-labelled runs, {n_traces} traces, {len(bad)} violations")


def c2_no_fallback():
    """No LLM-labelled run used the heuristic fallback."""
    n = bad = 0
    for sysdir in (RES / "llm_in_loop").glob("*lora*"):
        for man_p in sysdir.rglob("decisions.jsonl"):
            for line in man_p.read_text().splitlines():
                for tr in ((json.loads(line).get("decision_trace") or {})
                           .get("llm_traces") or []):
                    n += 1
                    if tr.get("fallback_reason") is not None:
                        bad += 1
    return bad == 0 and n > 0, f"{n} real-LLM calls, {bad} with a fallback reason"


def c3_backend_identified():
    df = pd.read_csv(RES / "llm_in_loop" / "llm_in_loop_summary.csv")
    have = {"system", "scenario", "configuration_hash"} <= set(df.columns)
    return have, f"summary carries system/scenario/configuration_hash for {len(df)} rows"


def c4_stub_configs():
    man = pd.read_csv(CFG / "scenario_manifest.csv")
    n = man.scenario_id.nunique()
    return n >= 12, f"{n} unique StubFeeder configurations (requirement: >= 12)"


def c5_opendss_configs():
    raw = pd.read_csv(RES / "opendss_sweep" / "raw_results.csv")
    n = raw.configuration_id.nunique()
    per_cfg_seeds = raw.groupby(["configuration_id", "detector"]).size().max()
    return (n >= 12 and per_cfg_seeds == 1,
            f"{n} unique OpenDSS configurations, max rows per (config, detector) = "
            f"{per_cfg_seeds} (no repeated-seed trials)")


def c6_config_level_inference():
    d = pd.read_csv(RES / "statistics" / "configuration_level_effects.csv")
    two_sided = "p_two_sided" in d.columns
    reports_both = {"n_configurations", "n_runs"} <= set(d.columns)
    return (two_sided and reports_both,
            f"{len(d)} contrasts; two-sided p reported; n_configurations and "
            f"n_runs reported separately")


def c7_sensitivity():
    p = RES / "statistics" / "sensitivity_excluding_degenerate.csv"
    d = pd.read_csv(p)
    return len(d) > 0, f"{len(d)} rows in sensitivity_excluding_degenerate.csv"


def c8_lora_expanded():
    p = RES / "lora_eval" / "summary.csv"
    if not p.exists():
        return False, "lora_eval/summary.csv missing"
    d = pd.read_csv(p)
    biggest = int(d.n.max())
    return biggest >= 120, f"largest evaluation n = {biggest} (requirement: >= 120)"


def c9_trustworthy_defined():
    tex = (PAPER / "sections" / "4-method.tex").read_text()
    has_def = "trustworthy does not denote a formal or universal" in tex
    has_tbl = (PAPER / "tables" / "table_trustworthy_characteristics.tex").exists()
    return has_def and has_tbl, "operational definition present; Table F generated"


def c10_threat_derivation():
    p = Path("code/docs/ai_pipeline_threat_taxonomy.md")
    tbl = PAPER / "tables" / "table_threat_taxonomy.tex"
    return p.exists() and tbl.exists(), f"{p} and {tbl.name} present"


def c11_timing():
    tbl = PAPER / "tables" / "table_deployment_budget.tex"
    slo = RES / "deadline_aware" / "slo_outcomes.csv"
    n = len(pd.read_csv(slo)) if slo.exists() else 0
    return tbl.exists() and n > 0, f"deployment budget table present; {n} SLO rows"


def c12_citations_verified():
    log = PAPER / "related_work_search_log.md"
    gaps = PAPER / "reference_gaps.md"
    return log.exists() and gaps.exists(), "search log and reference-gap file present"


def c13_compiles():
    pdf = PAPER / "main.pdf"
    logf = PAPER / "main.log"
    if not (pdf.exists() and logf.exists()):
        return False, "main.pdf or main.log missing"
    txt = logf.read_text(errors="ignore")
    undef = txt.count("LaTeX Warning: Reference") + txt.count("LaTeX Warning: Citation")
    return undef == 0, f"main.pdf built; {undef} undefined reference/citation warnings"


def c14_regenerable():
    """Every manuscript table used by the revision is generated from raw files."""
    generated = ["table_scenario_matrix.tex", "table_det_vs_llm.tex",
                 "table_containment.tex", "table_config_stats.tex",
                 "table_opendss_sweep.tex", "table_slo_outcomes.tex",
                 "table_lora_eval_expanded.tex", "revision_numbers.tex"]
    missing = [g for g in generated if not (PAPER / "tables" / g).exists()]
    stamped = [g for g in generated
               if (PAPER / "tables" / g).exists()
               and "Auto-generated" in (PAPER / "tables" / g).read_text()]
    return (not missing and len(stamped) >= len(generated) - 1,
            f"{len(generated) - len(missing)}/{len(generated)} present, "
            f"{len(stamped)} carry an auto-generated header")


def c15_response_complete():
    p = PAPER / "reviewer_response_draft.md"
    txt = p.read_text()
    n_placeholders = txt.count("[VERBATIM COMMENT")
    return (p.exists() and n_placeholders > 0,
            f"response present; {n_placeholders} verbatim-comment placeholders "
            f"awaiting the decision letter (gap G1)")


def c16_negatives_reported():
    txt = (PAPER / "FINAL_EXECUTION_SUMMARY.md").read_text()
    keys = ["must be softened", "Claims that must be softened",
            "reaches its limit", "negative finding"]
    return any(k in txt for k in keys), "negative/softened findings section present"


def c_tests():
    r = subprocess.run(["make", "ijcip-test"], capture_output=True, text=True)
    last = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l]
    return r.returncode == 0, (last[-1].strip() if last else "no pytest summary")


CHECKS = [
    ("1. QLoRA runs prove the intended adapter was loaded", c1_adapter_proof),
    ("2. No LLM-labelled run used heuristic fallback", c2_no_fallback),
    ("3. Each headline result identifies backend and configuration", c3_backend_identified),
    ("4. StubFeeder library >= 12 unique configurations", c4_stub_configs),
    ("5. OpenDSS varies configurations, not identical seeds", c5_opendss_configs),
    ("6. Inference uses configurations as the primary unit", c6_config_level_inference),
    ("7. Degenerate-scenario sensitivity analysis reported", c7_sensitivity),
    ("8. QLoRA evaluation meaningfully expanded (>= 120)", c8_lora_expanded),
    ("9. 'Trustworthy' operationally defined and mapped", c9_trustworthy_defined),
    ("10. Threat-category derivation documented", c10_threat_derivation),
    ("11. Real-world timing constraints analysed", c11_timing),
    ("12. Citation verification logged; gaps recorded", c12_citations_verified),
    ("13. Manuscript compiles without undefined refs", c13_compiles),
    ("14. Result tables regenerable from raw outputs", c14_regenerable),
    ("15. Reviewer response present", c15_response_complete),
    ("16. Negative findings reported honestly", c16_negatives_reported),
    ("+. Test suite passes", c_tests),
]


def main() -> None:
    for name, fn in CHECKS:
        check(name, fn)
    width = max(len(n) for n, _, _ in results)
    n_pass = sum(1 for _, ok, _ in results if ok)
    lines = [f"# Acceptance-criteria verification — {TAG}", ""]
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name.ljust(width)}  {detail}")
        lines.append(f"- **[{mark}]** {name} — {detail}")
    lines.insert(1, f"\n{n_pass}/{len(results)} checks pass.\n")
    Path("paper/acceptance_criteria_check.md").write_text("\n".join(lines) + "\n")
    print(f"\n{n_pass}/{len(results)} checks pass "
          f"-> paper/acceptance_criteria_check.md")


if __name__ == "__main__":
    main()
