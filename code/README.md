# code/ — DER-SafeAgent implementation (legacy package name: DER-SecAgent)

This package is used **from the repository root** as module entry points
(`python3 -m code.<module>`); it is deliberately not installed into
site-packages (see the layout note in `pyproject.toml`). The package and
directory names retain the legacy system name to keep the recorded
configuration hashes valid.

| Directory | Contents |
|---|---|
| `Multi_AI_Agent/` | Runtime path: runtime-safe FeatureView, Evidence Gate, safety-projection shield, class-avoidance map, energy/horizon estimators, coordinator, hash-chained audit record, prompts, synthetic worked examples, safety-invariant tests |
| `simulation/` | StubFeeder + OpenDSS harness, attack injectors, HITL model, scenario libraries (`scenarios/`), portability helpers |
| `llm_serving/` | Strict local QLoRA serving (`local_lora.py`), base-model resolution (`model_paths.py`) |
| `baselines/` | Deterministic energy policy (D0), single-LLM baseline, detector baselines |
| `evaluation/` | Experiment drivers. **`final_safeagent_v3/` is the canonical pipeline** behind the manuscript's main-text results; `ijcip_final_v3/` holds the forensics/property-test drivers; older directories (`final_safeagent/`, `final_revision/`, `llm_in_loop/`, …) are earlier-revision pipelines retained because canonical claim-audit data and the supplement reference them |
| `finetuning/` | QLoRA training recipes (`configs/`, `train.py`) and the released adapters under `results/*/adapter/` (Git LFS) |
| `finetuning_dataset/` | Synthetic 200-example corpus: builder, schema, validator, data card, processed splits with per-row SHA-256 |
| `configs/` | Hash-frozen protocol manifests (byte-exact; see `docs/DATA_PROVENANCE.md`) |
| `datasets/` | External-benchmark loaders + synthetic mock fixture (no third-party data redistributed) |
| `results/` | Canonical raw/processed results behind the paper (checksummed; per-run trace dumps excluded) |
| `docs/` | Threat model, AI-pipeline threat taxonomy, legacy reproducibility notes |
