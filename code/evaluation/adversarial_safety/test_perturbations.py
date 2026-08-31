"""Unit tests for the adversarial perturbation generators."""
from __future__ import annotations

from .perturbations import GENERATORS, build_suite


def test_every_family_generates_at_least_one_case():
    suite = build_suite(seed=0, n_per_family=3)
    seen = {c.perturbation for c in suite}
    assert seen == set(GENERATORS.keys())


def test_expected_behavior_keys_are_legal():
    legal = {"safe_actions", "must_not_action", "should_abstain",
              "schema_must_validate", "max_tier"}
    suite = build_suite(seed=0, n_per_family=2)
    for c in suite:
        for k in c.expected_behavior:
            assert k in legal, f"unknown key {k} on {c.case_id}"


def test_memory_poisoning_emits_override_record():
    suite = build_suite(seed=0, n_per_family=2)
    poisoned = [c for c in suite if c.perturbation == "memory_poisoning"]
    assert poisoned
    for c in poisoned:
        assert c.memory_overrides
        assert c.memory_overrides[0].chosen_action == "isolate_inverter"


def test_ood_attack_uses_out_of_taxonomy_label():
    legal_classes = {"none", "fdi", "replay", "command_spoof", "dos", "firmware"}
    suite = build_suite(seed=0, n_per_family=4)
    for c in [x for x in suite if x.perturbation == "ood_attack"]:
        for ev in c.events:
            lbl = ev.payload.get("ood_class")
            assert lbl is not None
            assert lbl not in legal_classes


def test_seed_is_deterministic():
    a = build_suite(seed=7, n_per_family=4)
    b = build_suite(seed=7, n_per_family=4)
    assert [c.case_id for c in a] == [c.case_id for c in b]
