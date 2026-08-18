# data/ — manifests and acquisition notes

This directory holds the **release manifests**; the datasets themselves live
in their pipeline-native locations under `code/` (kept there so the frozen
configuration hashes and module paths stay valid — see the repository
README's naming note).

## Manifests (`data/manifests/`)

| File | Contents |
|---|---|
| `release_manifest.json` | Curated per-artifact manifest: `artifact_id`, `relative_path`, `description`, `source`, `generation_script`, `configuration`, `split`, `seed`, `sha256`, `paper_reference`, `license`, `redistribution_status` for every canonical configuration, dataset, adapter, raw result, and reference artifact (60 entries). |
| `sha256_all.json` | SHA-256 of every canonical data file in the release (147 files). |

Verify everything:

```bash
make verify        # scripts/verify_artifacts.py — non-zero exit on any mismatch
```

Maintainers regenerate manifests **only** when the canonical artifact set
legitimately changes: `python3 scripts/make_release_manifests.py` (see the
warning in that script's docstring).

## Where the data lives

| Data | Location | Released? |
|---|---|---|
| Synthetic fine-tuning corpus (200 examples, 161/21/18) | `code/finetuning_dataset/processed/` (+ per-row digests in `manifest.json`) | yes (CC0-1.0, fully synthetic) |
| Leakage-controlled eval corpus (146 prompts) | `code/finetuning_dataset/revision_eval/eval_corpus.jsonl` | yes |
| Frozen protocol manifests (holdout, gate, estimator, OpenDSS subset) | `code/configs/` | yes (byte-exact, hash-pinned) |
| Scenario libraries (dev 25-config, holdout 49-config) | `code/simulation/scenarios/` | yes |
| Canonical raw/processed results | `code/results/` | yes (~2 MB; per-run trace dumps excluded) |
| QLoRA adapters | `code/finetuning/results/*/adapter/` | yes (Git LFS) |
| Mock external-alert fixture | `code/datasets/unsw_mg24/raw/alerts_mock.csv` | yes (synthetic) |
| UNSW-MG24 | — | **no** (research-data agreement); see `code/datasets/unsw_mg24/INSTALL_DATASET.md`: obtain from the UNSW authors' published distribution, place the alert CSV at `code/datasets/unsw_mg24/raw/alerts.csv` or set `UNSW_MG24_ROOT`. |
| TON_IoT | — | **no**; obtain from the UNSW TON_IoT distribution, set `TON_IOT_ROOT`. |
| Qwen / Llama base weights | — | **no** (model licenses); see `docs/MODEL_PROVENANCE.md`. |

Full provenance discussion: `docs/DATA_PROVENANCE.md`.
