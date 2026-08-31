"""Single-source-of-truth checks for the action policy.

The executable policy lives in code/Multi_AI_Agent/safety_projection.py.
scripts/export_action_policy.py derives from it (deterministically):

  * code/configs/action_policy.yaml       (machine-readable export)
  * artifacts/tables/table_action_policy.tex  (manuscript Table, tab:action-policy)

These tests fail if any exported view drifts from the runtime constants, or
if the adversarial suite's must-not sets fall outside the registry.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.Multi_AI_Agent.safety_projection import (  # noqa: E402
    IRREVERSIBLE, REGISTRY, project)


def _load_export():
    spec = importlib.util.spec_from_file_location(
        "export_action_policy", ROOT / "scripts/export_action_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_yaml_export_is_current():
    mod = _load_export()
    on_disk = (ROOT / "code/configs/action_policy.yaml").read_text()
    assert on_disk == mod.build_yaml(), (
        "code/configs/action_policy.yaml is stale — regenerate with "
        "python3 scripts/export_action_policy.py")


def test_manuscript_table_export_is_current():
    mod = _load_export()
    # The reference copy is tracked and must always match; the generated
    # copy exists only after `make reproduce-paper` (artifacts/tables/ is
    # not tracked) and is checked when present.
    ref = ROOT / "artifacts/reference/tables/table_action_policy.tex"
    assert ref.read_text() == mod.build_tex(), (
        "reference table is stale — regenerate with "
        "python3 scripts/export_action_policy.py and update the reference")
    gen = ROOT / "artifacts/tables/table_action_policy.tex"
    if gen.exists():
        assert gen.read_text() == mod.build_tex()


def test_model_proposed_irreversible_never_executes():
    # Exhaustive over the exported evidence classes and a confidence sweep:
    # a model-proposed irreversible primitive must never be the executed
    # action, whatever the class admits (escalated where admitted, vetoed
    # with deterministic fallback where class-inappropriate).
    mod = _load_export()
    for cls in mod.EVIDENCE_CLASSES:
        for conf in (0.0, 0.39, 0.41, 0.71, 1.0):
            for sev in (0.0, 0.31, 1.0):
                r = project(cls, "isolate_inverter", conf, sev)
                assert r.executed_action not in IRREVERSIBLE, (cls, conf, sev)
                if r.executed_action != "no_op":
                    assert r.source == "deterministic_fallback"


def test_adversarial_must_not_sets_within_registry():
    # The canonical 70-case suite (seed 0) declares its per-case must-not
    # sets; every entry must name a registry action, and the irreversible
    # primitive must be in every attack case's must-not set (that is the
    # quantity the paper's irreversible-execution metric counts).
    from code.evaluation.adversarial_safety.perturbations_expanded import (
        build_expanded_suite)
    cases = build_expanded_suite(n_per_family=5, seed=0)
    assert len(cases) == 70
    n_with_must_not = 0
    for c in cases:
        must_not = set(c.expected_behavior.get("must_not_action", []))
        assert must_not <= set(REGISTRY), (c.name, must_not - set(REGISTRY))
        if must_not:
            n_with_must_not += 1
            assert IRREVERSIBLE <= must_not or not (IRREVERSIBLE & must_not)
    assert n_with_must_not > 0


def test_fallback_never_selects_irreversible():
    mod = _load_export()
    for cls, order in mod.FALLBACK_ORDER.items():
        assert not (set(order) & IRREVERSIBLE), cls
