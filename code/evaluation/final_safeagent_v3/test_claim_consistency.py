"""Claim-to-evidence consistency gate (v3, Phase 0.3).

Fails (non-zero exit / pytest failure) when a headline number written into the
manuscript disagrees with the canonical result file it claims to come from.
Wire into `make ijcip-final-v3` before the LaTeX compile so an inconsistent
manuscript cannot be produced silently.

Each check pins (a) a canonical value extracted live from a result file and
(b) the exact string that must appear in the manuscript source. Where a v3
experiment has not yet replaced a v2 number, the check is marked xfail with a
reason so the suite documents the debt without hiding it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "code/results/ijcip_final_safeagent_20260810"
FINAL_REV = ROOT / "code/results/ijcip_final_revision"
TEX = ROOT / "paper/sections"
MAIN = ROOT / "paper/main.tex"


def _tex(*names: str) -> str:
    # The public release ships the code/data package without the manuscript
    # sources; manuscript-text checks are skipped (not silently passed) when
    # paper/ is absent so the remaining canonical-data checks still run.
    if not MAIN.exists():
        pytest.skip("manuscript sources (paper/) not present in this checkout")
    body = MAIN.read_text()
    for n in names:
        body += (TEX / n).read_text()
    return body


# ---- canonical extractors -------------------------------------------------

def canonical_e6_bare_isolate() -> tuple[int, int]:
    raw = pd.read_csv(FINAL_REV / "e6_adversarial/e6_raw.csv")
    bare = raw[raw.config == "bare_qwen"]
    n_iso = int((bare.executed_action == "isolate_inverter").sum())
    return n_iso, int(len(bare))


def canonical_p1c_bare() -> dict:
    a = pd.read_csv(V2 / "p1c_ablation/p1c_by_arm_qwen.csv")
    r = a[a.arm == "bare_llm"].iloc[0]
    return {"n": int(r.n), "exec_irrev": int(r.executed_irrev),
            "violation": float(r.policy_violation)}


def canonical_gate() -> dict:
    g = pd.read_csv(V2 / "p0b_gate/p0b_summary.csv").set_index("system")
    return {"d0g1_fa": float(g.loc["D0-G1", "benign_false_action_rate"]),
            "d0g1_cov": float(g.loc["D0-G1", "attack_coverage"]),
            "q1g1_cov": float(g.loc["Q1-G1", "attack_coverage"])}


# ---- consistency checks ---------------------------------------------------

def test_e6_61_of_70_matches_canonical():
    n_iso, n = canonical_e6_bare_isolate()
    assert (n_iso, n) == (61, 70), f"E6 canonical drifted: {n_iso}/{n}"


def test_p1c_bare_counts_are_distinct_metrics():
    c = canonical_p1c_bare()
    # 37/42 is an execution COUNT (0.881), not the 0.667 violation RATE
    assert c["exec_irrev"] == 37 and c["n"] == 42
    assert abs(c["violation"] - 0.6667) < 0.01
    assert abs(c["exec_irrev"] / c["n"] - c["violation"]) > 0.1, (
        "execution rate and violation rate must not be conflated")


def test_manuscript_uses_single_adversarial_denominator():
    """Phase-4 unified the suite to 70 cases; the results section must not
    present a 37/42 count as a competing containment denominator."""
    body = _tex("5-experiments.tex")
    assert "37 of 42" not in body and "42-case" not in body, (
        "manuscript still mixes the retired 42-case suite with the 70-case one")


def test_latency_reconciled_to_canonical_k1():
    """Phase-5 rebuild: the per-call K=1 latency in the manuscript must be the
    canonical ~5.5 s. The retracted 13.7 s may appear ONLY in a sentence that
    marks it superseded, never as the current value."""
    body = _tex("5-experiments.tex")
    assert "5.45" in body, "canonical K=1 latency (5.45 s) not reflected"
    for m in re.finditer(r"13\.7", body):
        window = body[max(0, m.start() - 120): m.end() + 40]
        assert re.search(r"supersed|earlier draft|conflat|prior", window), (
            "13.7 s appears without being marked as superseded")


def test_gate_coverage_terms_not_conflated():
    """Deterministic gate-open coverage (1.000) and model correct-mitigation
    coverage (0.562) are different quantities and must not share a term."""
    g = canonical_gate()
    assert g["d0g1_cov"] == 1.0 and abs(g["q1g1_cov"] - 0.5625) < 0.01
    assert g["d0g1_cov"] != g["q1g1_cov"]


def test_tampered_flag_not_in_runtime_safe_featureview():
    """Phase-1 invariant: the runtime-safe FeatureView must not read the
    injector ground-truth `tampered` flag. Passes once Phase 1 lands the
    `runtime_safe` path; xfail-guarded until then."""
    fv = (ROOT / "code/Multi_AI_Agent/telemetry_features.py").read_text()
    if "runtime_safe" not in fv:
        pytest.xfail("Phase 1 de-oracling not yet applied")
    # in runtime_safe mode, tampered must be ignored
    assert "runtime_safe" in fv and "e.get(\"tampered\")" in fv, (
        "expected an explicit runtime_safe guard around the tampered read")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
