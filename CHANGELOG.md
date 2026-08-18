# Changelog

## Unreleased

### Fixed
- `scripts/smoke_test.sh`, `scripts/reproduce_paper_artifacts.sh`, and
  `scripts/reproduce_all.sh` now honor the `PY` environment variable
  (previously hard-coded `python3`), and the `Makefile` exports `PY` to
  recipe subshells, so `make smoke PY=/path/to/python` selects the intended
  interpreter end-to-end. No scientific content changed; the full
  verification ladder (`make test` / `smoke` / `verify` /
  `reproduce-paper`) was re-run and passed after the change (2026-08-17).

## 1.0.0 — 2026-08-16 — initial public reproducibility release

First public packaging of the DER-SafeAgent research tree (internal tag
`ijcip_final_v3`, experiments executed 2026-08-11/12). The scientific
content — frozen protocols, canonical raw results, statistics, and the
values in every regenerated table/figure — is byte-for-byte identical to the
material behind the submitted manuscript. Changes made for public release
are packaging-only:

### Added
- Top-level reproducibility scaffolding: `Makefile`, `scripts/` (bootstrap,
  smoke test, artifact verification, paper-artifact regeneration and
  comparison, latency re-derivation, environment capture, manifest
  generator), `tests/` (repo smoke, manifest integrity, paper-artifact
  regeneration), `docs/`, CI workflow, Dockerfile, `pyproject.toml`,
  `requirements(-lock).txt`, `environment.yml`.
- `data/manifests/release_manifest.json` and `sha256_all.json`: SHA-256
  manifests over every canonical configuration, dataset, adapter, and result
  artifact.
- `artifacts/reference/`: byte-exact copies of the manuscript's table and
  figure files, used as the comparison target for regeneration.
- `code/simulation/portable_paths.py`: load-time remapping of the absolute
  scenario paths recorded inside the hash-frozen manifests (the frozen CSV
  bytes are unchanged so their recorded digests stay valid).
- `code/llm_serving/model_paths.py`: base-model resolution via
  `DER_QWEN_BASE` / `DER_LLAMA_BASE` env vars, defaulting to the Hugging
  Face IDs.
- `scripts/derive_latency_v3.py`: documents and verifies that the canonical
  latency summary (Table 10) re-derives value-identically from the archived
  strict-serving prediction logs.

### Changed (packaging-only; numbers unchanged, verified byte-identical)
- `code/evaluation/final_safeagent_v3/build_artifacts_v3.py`: outputs now go
  to `artifacts/` (the repository has no `paper/` tree); row/backend display
  labels and the Table 9 caption now match the manuscript exactly, so
  regenerated tables are byte-identical to the submitted ones (previously
  the manuscript copies carried cosmetic label edits applied after
  generation).
- Hard-coded machine-local base-model paths in the LLM drivers and
  fine-tuning configs replaced with the Hugging Face model IDs / env-var
  override; `adapter_config.json` `base_model_name_or_path` likewise (weights
  and adapter SHA-256 digests untouched).
- `test_claim_consistency.py`: manuscript-text checks skip (not pass) when
  the `paper/` tree is absent from a checkout.
- Machine-local absolute path prefixes removed from non-hash-pinned
  provenance copies (`run_manifest.json`, OpenDSS scenario snapshots,
  `scenario_matrix.yaml`, `config_resolved.yaml`). The hash-pinned frozen
  manifests keep their original bytes; see `docs/DATA_PROVENANCE.md`.

### Excluded from the public tree (with acquisition/provenance docs)
- UNSW-MG24 (8.4 GB) and TON_IoT (69 MB) third-party datasets — not
  redistributable; loaders, mock fixture, and acquisition instructions ship
  instead (`data/README.md`).
- Copyrighted background-reading PDFs, per-run trace dumps (463 MB
  `holdout_e2e/runs/`), training checkpoints/optimizer states, logs, and the
  unimplemented StackStorm stub.
