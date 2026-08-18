# Data Provenance

Every dataset, frozen manifest, and canonical result file in this release is
listed with its SHA-256 in `data/manifests/release_manifest.json` (curated,
with per-artifact metadata) and `data/manifests/sha256_all.json` (complete
digest listing). `make verify` re-checks all of them plus the internal
freeze-JSON pins. This document explains where each class of data came from.

## 1. Synthetic fine-tuning corpus (released)

`code/finetuning_dataset/` — 200 ChatML examples, split 161/21/18
(train/val/test), built deterministically from a synthetic shard with
`build_dataset.py --seed 0` and schema-validated by `validate.py`
(`schema.py`, pydantic). 100% synthetic: every row carries
`metadata.source = "synthetic"`, `license = CC0-1.0`; identifiers, IPs, and
assets are invented (IEEE 13/34-bus node names). Per-row SHA-256 digests in
`processed/manifest.json`; corpus card in `DATA_CARD.md`. No PII, no
real-world captures, no third-party text.

The leakage-controlled 146-prompt evaluation corpus
(`revision_eval/eval_corpus.jsonl`) is likewise synthetic/scenario-derived;
the adversarial driver additionally re-checks at run time that no test case
duplicates a training example (exact or normalized;
`code/results/ijcip_final_v3/adversarial/leakage_summary.json`).

## 2. Frozen experiment protocols (released, byte-exact)

| Asset | File(s) | Freeze record |
|---|---|---|
| 49-config end-to-end holdout (33 attack / 16 benign; episode seeds 1301–1303) | `code/configs/ijcip_final_v3/holdout_v3_manifest.csv` + 49 scenario YAMLs under `code/simulation/scenarios/ijcip_final_v3/` | `holdout_v3_freeze.json` |
| Evidence-Gate operating point (p_hard=2, soft disabled, z_soft=3.0) | `code/configs/ijcip_final_v3/evidence_gate_v3_frozen.json` (self-hashing) | itself |
| 32-config gate test set | `code/configs/ijcip_final_safeagent_20260810/evidence_gate_test.csv` | `evidence_gate_freeze.json` |
| Gate dev-benign set | `evidence_gate_dev_benign.csv` | — (dev split) |
| 12-config estimator holdout + duration prior | `impact_estimator_holdout.csv`, `eh_duration_prior.json` | `impact_estimator_freeze.json` |
| 25-config development scenario library | `code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv` + scenario YAMLs | `scenario_matrix.yaml` |
| Pre-registered 12-config OpenDSS subset | `opendss_llm_subset.csv` | `opendss_llm_subset_freeze.json` |

**Absolute-path note (deliberate, documented exception).** The frozen
manifest CSVs record the `config_path` of each scenario as an absolute path
on the original experiment machine. Rewriting those bytes would invalidate
the SHA-256 digests recorded in the freeze files, so the CSVs are shipped
byte-exact and the consumers remap paths at load time
(`code/simulation/portable_paths.py`). Consequence: the original machine's
filesystem prefix (including a username) is visible inside these frozen
manifests. This is a provenance trade-off, not an accident;
`tests/test_manifests.py::test_no_machine_local_paths_outside_frozen_manifests`
enforces that no *other* file leaks machine-local paths. Non-hash-pinned
provenance copies (`run_manifest.json`, OpenDSS scenario snapshots,
`config_resolved.yaml`) had their local prefixes rewritten to portable
equivalents at packaging time (see `CHANGELOG.md`).

## 3. Canonical results (released)

All result files behind the manuscript's tables/figures, ~2 MB total —
see `docs/PAPER_ARTIFACT_MAP.md` for the item-by-item mapping and
`data/manifests/release_manifest.json` for per-file provenance metadata
(generation script, configuration, seeds, digest). Includes the runtime-side
provenance categories: raw LLM proposals and parsed decisions (per-row in
the adversarial raw CSVs; per-run `decisions.jsonl` traces are in the
excluded bulk dump, see §5), shield vetoes / deterministic fallbacks /
escalations / executed actions (columns of `holdout_e2e_raw.csv` and the
adversarial raw CSVs), latency measurements (`lora_eval/*/predictions.jsonl`
+ derived summaries), structural-enumeration and mutation-test results
(`property_tests/property_test_result.json`), and gate-invariance forensics
(`p0_forensics/*.json`).

## 4. Third-party datasets (NOT redistributed)

| Dataset | Role in the paper | Status in this release |
|---|---|---|
| UNSW-MG24 | external-benchmark adapter scaffolding (not part of the principal results) | **not included** (research-data agreement). The loader (`code/datasets/unsw_mg24/loader.py`), a synthetic mock fixture (`raw/alerts_mock.csv`), and acquisition instructions (`INSTALL_DATASET.md`) ship instead. `production_loader.py` tags every sample `result_source ∈ {production, mock}` so mock-derived rows can never pass as real-data results. Expected drop-in: `code/datasets/unsw_mg24/raw/alerts.csv`. |
| TON_IoT | same (adapter scaffolding) | **not included**; loader in `code/datasets/ton_iot/`, root via `TON_IOT_ROOT`. |
| OpenDSS IEEE 13/34-bus test feeders | physical-fidelity power-flow check | feeder definitions ship with the simulation harness (public IEEE test cases; upstream: the OpenDSS distribution). IEEE-123 is not included — the paper states the reduced feeder scope explicitly. |

## 5. Excluded from the public tree (and why)

- `holdout_e2e/runs/` per-run trace dumps (463 MB): aggregated into the
  released `holdout_e2e_raw.csv`; the forensics JSONs computed from them are
  released with digests. Re-running `make holdout-cpu` / `make holdout-llm`
  regenerates traces locally.
- UNSW-MG24 (8.4 GB) / TON_IoT (69 MB) raw data: license (see §4).
- Background-reading PDFs that sat next to the corpus builder: copyrighted,
  unused by any pipeline.
- Training checkpoints / optimizer states: 314 MB of redundancy; the
  canonical adapters are released.
- Internal revision-process notes and logs.

## 6. Ground-truth separation (the leakage question)

Attack-generator ground truth (`tampered` flags, injected-attack metadata)
exists only inside scenario configs and simulator events. The runtime path —
FeatureView → Evidence Gate → shield → execution — runs with
`runtime_safe=True` and is **proven** unable to read it:
`test_runtime_safe_gate.py` flips every ground-truth flag and asserts the
runtime FeatureView is bit-for-bit unchanged; metrics are computed offline
after the episode ends. The single deliberate exception is the **OPROJ**
oracle arm, which exists as a labelled upper bound and is excluded from
gate-invariance checks. See Table 2 of the paper (provenance/trust
boundary), which this test suite enforces mechanically.
