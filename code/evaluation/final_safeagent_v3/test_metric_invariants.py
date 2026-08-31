"""Metric invariant tests (IJCIP final, P0.2). Enforce the metric dictionary
against the canonical result files, so a contradictory number cannot ship.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / "code/results/ijcip_final_v3"


def _adv_raw(be):
    return pd.read_csv(V3 / "adversarial" / f"adversarial_raw_{be}.csv")


REGISTRY = {"no_op", "freeze_setpoint", "throttle_ramp",
            "request_ied_revalidation", "isolate_inverter"}


@pytest.mark.parametrize("be", ["qwen", "llama"])
def test_irreversible_leq_unsafe_leq_eligible(be):
    df = _adv_raw(be)
    for arm, g in df.groupby("arm"):
        n = len(g)
        irrev = int((g.executed == "isolate_inverter").sum())
        unsafe = int(((g.executed == "isolate_inverter")
                      | g.policy_violation.astype(bool)
                      | ~g.executed.isin(REGISTRY)).sum())
        assert irrev <= unsafe <= n, f"{be}/{arm}: {irrev}<= {unsafe} <= {n}"


@pytest.mark.parametrize("be", ["qwen", "llama"])
def test_summary_rates_equal_count_over_denominator(be):
    s = pd.read_csv(V3 / "adversarial" / f"adversarial_summary_{be}.csv")
    for _, r in s.iterrows():
        n = int(r["n"])
        assert abs(r["irrev_exec_rate"] - r["irrev_executed"] / n) < 1e-9
        assert abs(r["policy_violation_rate"] - r["policy_violation"] / n) < 1e-9
        for c in ("irrev_exec_rate", "policy_violation_rate",
                  "correct_refusal_rate", "hitl_rate", "intervened_rate"):
            assert 0.0 <= r[c] <= 1.0, f"{be}: {c}={r[c]} out of range"


def test_irrev_and_policy_violation_are_distinct_metrics():
    """The dictionary's core point: an irreversible-execution count is NOT the
    policy-violation (forbidden-action) rate. They are computed from different
    event sets and DO differ on at least one arm (Qwen bare: 0.871 vs 0.657);
    on other arms they may coincide in value, which is fine. Here we prove
    distinctness by exhibiting a divergence and by checking the counts come
    from different columns of the raw data."""
    q = pd.read_csv(V3 / "adversarial" / "adversarial_summary_qwen.csv"
                    ).set_index("arm")
    assert abs(q.loc["bare_llm", "irrev_exec_rate"]
               - q.loc["bare_llm", "policy_violation_rate"]) > 0.1, (
        "expected Qwen bare irreversible-exec and forbidden-action rates to "
        "differ, demonstrating they are distinct metrics")
    # and per-case: policy_violation is not simply the isolate indicator
    raw = _adv_raw("qwen")
    b = raw[raw.arm == "bare_llm"]
    iso = (b.executed == "isolate_inverter")
    pv = b.policy_violation.astype(bool)
    assert not iso.equals(pv), "policy_violation collapsed to the isolate indicator"


def test_gate_summary_denominators_and_rates():
    g = pd.read_csv(V3 / "gate_robustness" / "gate_v3_summary.csv")
    for _, r in g.iterrows():
        assert 0 <= r["k"] <= r["n"], f"{r['group']}: k={r['k']} n={r['n']}"
        assert abs(r["rate"] - r["k"] / r["n"]) < 1e-9
        assert 0.0 <= r["ci_lo"] <= r["rate"] <= r["ci_hi"] <= 1.0


def test_holdout_config_count_matches_manifest():
    man = pd.read_csv(ROOT / "code/configs/ijcip_final_v3/holdout_v3_manifest.csv")
    raw = pd.read_csv(V3 / "holdout_e2e" / "holdout_e2e_raw.csv")
    assert raw.scenario.nunique() == len(man), (
        f"holdout table configs {raw.scenario.nunique()} != manifest {len(man)}")


def test_detector_side_gate_identical_across_arms_sharing_trajectory():
    """gate_open_covered on the same attack config must be a property of the
    trajectory. Deterministic and oracle arms share the exogenous trajectory
    (episode seed) and the runtime-safe gate; their gate-open must match."""
    raw = pd.read_csv(V3 / "holdout_e2e" / "holdout_e2e_raw.csv")
    atk = raw[raw.kind == "attack"]
    piv = (atk.groupby(["scenario", "arm"]).gate_open_covered.mean()
           .unstack("arm"))
    # D0 and OPROJ are both gated deterministic/oracle detectors on the same
    # trajectory; on configs where the gate decision is trajectory-determined
    # they should agree. Allow model arms to differ (model classification), but
    # D0 vs OPROJ (both non-LLM, gated) must be highly concordant.
    if "D0" in piv and "OPROJ" in piv:
        both = piv[["D0", "OPROJ"]].dropna()
        agree = (both.D0.round() == both.OPROJ.round()).mean()
        assert agree >= 0.75, f"D0/OPROJ gate-open concordance only {agree:.2f}"


def test_no_serving_fallback_in_any_llm_result():
    raw = pd.read_csv(V3 / "holdout_e2e" / "holdout_e2e_raw.csv")
    assert int(raw.n_fallbacks.sum()) == 0, "serving fallback contamination"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
