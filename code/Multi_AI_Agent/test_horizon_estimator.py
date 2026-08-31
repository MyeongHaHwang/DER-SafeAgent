"""Unit tests for the horizon-aware estimator (P0-A)."""
from __future__ import annotations

from .horizon_estimator import (HorizonCalibration, class_family, estimate_eh,
                                estimate_emh, select)
from .telemetry_features import FeatureView


def _fv(**kw) -> FeatureView:
    base = dict(t=185.0, asset_p_kw={"INV_634": 30.0, "BESS_675": 105.0},
                asset_p_avail_kw={"INV_634": 200.0, "BESS_675": 150.0},
                severity_score=0.8)
    base.update(kw)
    return FeatureView(**base)


CAL = HorizonCalibration.from_durations(
    {"spoof_like": [60, 120, 300], "fdi_like": [120], "stale": [120, 300]})


def test_no_future_information_in_signature():
    import inspect
    for fn in (estimate_eh, estimate_emh):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"attack_end_s", "true_end_s", "end_s"}, \
            "estimator must not receive the true attack termination time"


def test_spoof_prefers_freeze_over_horizon():
    fv = _fv(n_command_events=3, dominant_signal="command")
    assert class_family(fv) == "spoof_like"
    est = estimate_eh(fv, "INV_634", 5.0, CAL, load_demand_kw=1000.0)
    assert select(est, family="spoof_like").action == "freeze_setpoint"


def test_fdi_prefers_no_op():
    fv = _fv(n_tampered_events=4, dominant_signal="tampered",
             asset_p_kw={"INV_634": 190.0, "BESS_675": 105.0})
    assert class_family(fv) == "fdi_like"
    est = estimate_eh(fv, "INV_634", 5.0, CAL, load_demand_kw=1000.0)
    assert select(est, family="fdi_like").action == "no_op"


def test_stale_family_prefers_freeze_minimax():
    fv = _fv(n_dup_events=4, dominant_signal="replay")
    assert class_family(fv) == "stale"
    est = estimate_eh(fv, "INV_634", 5.0, CAL, load_demand_kw=1000.0)
    assert select(est, family="stale").action == "freeze_setpoint"


def test_isolate_costed_over_restoration_horizon():
    fv = _fv(n_command_events=3, dominant_signal="command")
    est = {e.action: e for e in estimate_eh(fv, "INV_634", 5.0, CAL,
                                            load_demand_kw=1000.0)}
    # isolate accrues nominal output over 600 s regardless of incident horizon
    assert est["isolate_inverter"].expected_curt_kwh > \
        est["freeze_setpoint"].expected_curt_kwh * 10


def test_none_family_forces_no_op():
    fv = _fv(severity_score=0.1)
    est = estimate_eh(fv, "INV_634", 5.0, CAL, load_demand_kw=1000.0)
    assert select(est, family="none", severity=0.1).action == "no_op"


def test_emh_close_to_eh_linear_terms():
    fv = _fv(n_command_events=3, dominant_signal="command")
    eh = {e.action: e.expected_curt_kwh
          for e in estimate_eh(fv, "INV_634", 5.0, CAL, load_demand_kw=1000.0)}
    emh = {e.action: e.expected_curt_kwh
           for e in estimate_emh(fv, "INV_634", 5.0, CAL, load_demand_kw=1000.0)}
    for a in eh:
        assert abs(eh[a] - emh[a]) < 1e-6
