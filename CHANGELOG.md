# Changelog

## 1.1.0 — 2026-08-29 — release-candidate audit and manuscript sync

Full claim-by-claim audit against the manuscript (`docs/CLAIM_LEDGER.md`);
no raw result files were modified. One derived statistics protocol was
amended (below), and the manuscript was synchronized with the release.

### Changed (statistics re-analysis; no new experiments)
- `run_stats_v3.py`: the sign-flip permutation and bootstrap resampling
  draws are now a deterministic function of the family seed (20260811) and
  sample size, shared across all contrasts (common random numbers). The
  original 2026-08-11 run consumed one sequential RNG stream, so the
  byte-identical ModelOnly-Q/L difference vectors received different
  bootstrap samples and near-identical-but-unequal CIs in Table 9. Identical
  vectors now receive identical CIs; every qualitative conclusion (Holm
  significance pattern) is unchanged. `F1_physical.csv`, the Table 9
  reference, and the three body-text p-values were updated accordingly.

### Added
- `docs/CLAIM_LEDGER.md`: 35 principal manuscript claims traced to raw
  sources with executed verification commands and statuses.
- `scripts/export_action_policy.py` + `code/configs/action_policy.yaml` +
  manuscript Table 3 (`table_action_policy.tex`): single-source-of-truth
  export of the shield's class-aware action policy, guarded by
  `tests/test_action_policy.py` (5 tests, incl. an exhaustive
  model-proposed-irreversible sweep and adversarial must-not-set checks).
- `paper/`: the manuscript LaTeX source (main + supplementary + tables +
  figures + listings) is now tracked; `make latex` builds it, and the
  claim-consistency tests now run against it in CI instead of skipping.
- `code/results/ijcip_final_revision/e7_trustworthiness/`: the groundedness
  audit CSVs behind the 70–73% / 23–30% discussion figures (previously
  cited but not shipped).
- `scripts/repo_hygiene.py` + `make hygiene`: secret / stale-wording /
  absolute-path / oversize-file scan.
- `make release-check`: single CPU gate (deps, compile, tests, smoke,
  verify, reproduce-paper, latency, hygiene, LaTeX build when available).
- CI: hygiene step and a manuscript-build job.
- `DER-SafeAgent_Logo.png` and the README logo header (from the initial
  GitHub upload).

### Fixed / synchronized
- Manuscript: de-anonymized author block (IJCIP review is
  single-anonymized); Code and Data Availability now cites the verified
  repository URL; three verified IJCIP references added to Related Work and
  the positioning table; irreversible-action wording made exact (vetoed
  where class-inappropriate, escalated where admitted — matching
  `safety_projection.py`); explicit statement that RQ1 arms generate model
  output independently (no frozen-output replay); Table 3 insertion shifts
  the old Tables 3–10 to 4–11 (all repository docs renumbered).
- Removed stale double-blind wording from README, CITATION.cff, LICENSE,
  and SECURITY.md; CITATION.cff now carries the real author list and
  repository URL.
- Resolved the merge-conflicted LICENSE on the initial GitHub upload to
  **Apache-2.0** (author-confirmed) with the third-party scope note
  retained; CITATION.cff, pyproject.toml, README, and all release-manifest
  license fields updated consistently.

## 1.0.1 — 2026-08-17

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
