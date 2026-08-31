"""IJCIP P0 sensitive / ambiguous scenario library.

Each entry in :data:`SCENARIOS` is a single-step harness probe with an
engineered (telemetry, events) tuple plus an ``expected_behavior``
dict, designed so that removing one of the multi-agent components
produces a measurably different decision. The library is consumed by
``code/evaluation/ablation/run_sensitive_ablation.py``.
"""
from .scenarios import SCENARIOS, SensitiveScenario  # noqa: F401
