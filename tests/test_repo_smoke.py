"""Smoke tests: imports, frozen configurations, schemas, registry, corpus.

CPU-only, no GPU, no LLM, no OpenDSS. These are the fast checks behind
``make smoke`` (together with the safety-invariant test modules under
``code/``).
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "code.Multi_AI_Agent.evidence_gate",
    "code.Multi_AI_Agent.telemetry_features",
    "code.Multi_AI_Agent.safety_projection",
    "code.Multi_AI_Agent.energy_estimator",
    "code.Multi_AI_Agent.horizon_estimator",
    "code.Multi_AI_Agent.audit_chain",
    "code.simulation.feeder",
    "code.simulation.harness",
    "code.simulation.types",
    "code.simulation.portable_paths",
    "code.llm_serving.local_lora",
    "code.llm_serving.model_paths",
    "code.evaluation.final_safeagent_v3.build_artifacts_v3",
    "code.evaluation.final_safeagent_v3.run_stats_v3",
    "code.datasets.unsw_mg24.loader",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)


def test_frozen_gate_operating_point():
    """The frozen Evidence-Gate parameters used in the manuscript."""
    fz = json.loads(
        (ROOT / "code/configs/ijcip_final_v3/evidence_gate_v3_frozen.json")
        .read_text())
    assert fz["p_hard"] == 2
    assert fz["p_soft"] == 10000        # soft channel disabled at the frozen point
    assert fz["z_soft"] == 3.0
    assert fz["runtime_safe"] is True
    assert len(fz["sha256"]) == 64


def test_holdout_manifest_composition():
    """49 hash-frozen holdout configurations: 33 attack + 16 benign."""
    man = pd.read_csv(ROOT / "code/configs/ijcip_final_v3/holdout_v3_manifest.csv")
    assert len(man) == 49
    kinds = man.kind.value_counts().to_dict()
    assert kinds == {"attack": 33, "benign": 16}
    assert man.config_sha256.str.len().eq(64).all()


def test_holdout_manifest_paths_resolve():
    """Frozen manifests store machine-local paths; the portable remap works."""
    from code.simulation.portable_paths import resolve_config_path
    man = pd.read_csv(ROOT / "code/configs/ijcip_final_v3/holdout_v3_manifest.csv")
    for rec in man.config_path.head(5):
        p = resolve_config_path(rec)
        cfg = yaml.safe_load(p.read_text())
        assert "ground_truth" in cfg or "attack" in cfg or "feeder" in cfg


def test_action_registry_is_fixed():
    from code.Multi_AI_Agent.safety_projection import REGISTRY
    assert set(REGISTRY) == {"no_op", "freeze_setpoint", "throttle_ramp",
                             "request_ied_revalidation", "isolate_inverter"}


def test_finetuning_corpus_schema_and_split():
    """200-example synthetic corpus: 161/21/18 split, schema-valid rows."""
    from code.finetuning_dataset.schema import Example
    proc = ROOT / "code/finetuning_dataset/processed"
    sizes = {}
    for split in ("train", "val", "test"):
        rows = [json.loads(l) for l in (proc / f"{split}.jsonl").open()]
        sizes[split] = len(rows)
        for r in rows[:5] + rows[-5:]:
            Example(**r)                    # pydantic validation
            assert r["metadata"]["source"] == "synthetic"
    assert sizes == {"train": 161, "val": 21, "test": 18}


def test_mock_alert_synthesizer(tmp_path):
    from code.datasets.unsw_mg24.loader import synthesize_alerts
    out = tmp_path / "alerts.csv"
    synthesize_alerts(str(out), n_alerts=20, seed=0)
    df = pd.read_csv(out)
    assert len(df) == 20


def test_llm_serving_defaults_are_portable():
    """No absolute machine-local base-model paths in the release."""
    from code.llm_serving import model_paths
    assert not model_paths.QWEN_BASE.startswith("/home/")
    assert not model_paths.LLAMA_BASE.startswith("/home/")
    for cfg in (ROOT / "code/finetuning/results").glob("*/adapter/adapter_config.json"):
        base = json.loads(cfg.read_text())["base_model_name_or_path"]
        assert not base.startswith("/"), f"absolute base path in {cfg}"
