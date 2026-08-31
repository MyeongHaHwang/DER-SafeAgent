"""Energy-impact metrics for DER cybersecurity scenarios.

Each function takes a `pandas.DataFrame` representing a single scenario run's
time-series and returns a `MetricResult`. Inputs are expected to follow the
schema defined in `schema.md`. Functions are pure and side-effect free so they
can be unit-tested without the simulator.
"""
from .ens import energy_not_supplied
from .voltage import voltage_deviation
from .frequency import frequency_excursion
from .ramp import ramp_violations
from .curtailment import curtailed_energy
from .types import MetricResult, TimeSeries

__all__ = [
    "energy_not_supplied",
    "voltage_deviation",
    "frequency_excursion",
    "ramp_violations",
    "curtailed_energy",
    "MetricResult",
    "TimeSeries",
]
