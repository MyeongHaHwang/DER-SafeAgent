"""Unit tests for energy_metrics. Run with: pytest test_metrics.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from . import (
    curtailed_energy,
    energy_not_supplied,
    frequency_excursion,
    ramp_violations,
    voltage_deviation,
)


def _ts(n: int = 60, dt: float = 1.0) -> pd.DataFrame:
    t = np.arange(n) * dt
    return pd.DataFrame({"t": t})


def test_ens_zero_when_demand_met():
    ts = _ts()
    ts["load_demand"] = 100.0
    ts["load_served"] = 100.0
    assert energy_not_supplied(ts).value == pytest.approx(0.0)


def test_ens_simple_outage():
    ts = _ts(n=3601, dt=1.0)  # 1 h @ 1 Hz
    ts["load_demand"] = 100.0
    ts["load_served"] = 50.0
    # 50 kW deficit for 1 h ≈ 50 kWh
    assert energy_not_supplied(ts).value == pytest.approx(50.0, rel=0.01)


def test_voltage_deviation_within_band():
    ts = _ts()
    ts["v_pu_BUS1"] = 1.0
    res = voltage_deviation(ts)
    assert res.value == 0.0


def test_voltage_deviation_excursion():
    ts = _ts()
    ts["v_pu_BUS1"] = 1.07  # > 1.05
    assert voltage_deviation(ts).value == 1.0


def test_frequency_excursion_max():
    ts = _ts()
    ts["freq"] = 60.0
    ts.loc[5:10, "freq"] = 59.5
    res = frequency_excursion(ts)
    assert res.value == pytest.approx(0.5)


def test_ramp_violations_counts_steps():
    ts = _ts(n=10, dt=1.0)
    ts["p_pv_INV1"] = [0, 0, 0, 100, 100, 100, 100, 100, 100, 100]  # 100 kW/s step
    res = ramp_violations(ts, limit_kw_per_s=10.0)
    assert res.value == 1


def test_curtailment_zero_when_dispatched_at_avail():
    ts = _ts()
    ts["p_pv_INV1"] = 50.0
    ts["p_pv_avail_INV1"] = 50.0
    assert curtailed_energy(ts).value == pytest.approx(0.0)


def test_curtailment_reports_gap():
    ts = _ts(n=3601, dt=1.0)
    ts["p_pv_INV1"] = 0.0
    ts["p_pv_avail_INV1"] = 10.0
    # 10 kW gap for 1 h ≈ 10 kWh
    assert curtailed_energy(ts).value == pytest.approx(10.0, rel=0.01)
