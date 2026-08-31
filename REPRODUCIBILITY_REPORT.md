# Reproducibility Report — DER-SafeAgent public release 1.1.0

Verification record. Originally prepared 2026-08-16 (release 1.0.0),
re-verified 2026-08-17 (1.0.1, `PY` interpreter override), and fully
re-audited 2026-08-29 for release 1.1.0: claim-by-claim manuscript audit
(`docs/CLAIM_LEDGER.md`), amended Table 9 statistics protocol (shared
resampling draws; see CHANGELOG), manuscript source tracked under `paper/`
(claim-consistency tests now execute instead of skipping), action-policy
single-source export, and repository hygiene gate. 1.1.0 verification on
E1: `make test` 179 passed / 0 skipped; `make smoke` all 4 stages;
`make verify` 236 checks / 0 failed; `make reproduce-paper` 14 comparisons
passed (8 tables + 3 figure PNGs byte-identical, 3 statistics CSVs
hash-identical); `make hygiene` clean; manuscript rebuild 52 pp, no
undefined references. Environments:

- **E1 (reference)**: the machine that produced the canonical results —
  Linux 6.17, Python 3.12.3, RTX 3080 (10 GiB), CUDA 12.4, torch
  2.6.0+cu124, transformers 4.57.6, peft 0.14.0, bitsandbytes 0.49.2,
  numpy 2.4.3, pandas 3.0.2, matplotlib 3.10.9 (full pin:
  `requirements-lock.txt`; archived original:
  `code/results/ijcip_final_v3/environment.json`).
- **E2 (clean)**: a fresh copy of this repository in a new virtual
  environment with **only** `pip install -r requirements.txt` (CPU stack;
  resolved at verification time to numpy 2.5.2, pandas 3.0.5,
  matplotlib 3.10.9). No GPU stack installed.

Statuses: **PASS** = executed and verified; **PASS (E1 only)** = verified on
the reference machine, not repeatable without GPU/base weights;
**NOT RE-RUN** = shipped artifact is checksummed but the producing
computation was not re-executed for this release.

| Component | Command | Environment | Status | Evidence | Limitation |
|---|---|---|---|---|---|
| Dependency install (CPU core) | `pip install -r requirements.txt` | E2 | PASS | clean venv resolved and installed; all subsequent E2 rows ran on it | GPU extras (`make setup-llm`) not installed in E2 |
| Full test suite (179 tests) | `make test` | E1 + E2 | PASS | 1.1.0 (E1): `179 passed, 0 skipped` — the manuscript-text checks now execute against the tracked `paper/` source, plus the 5 new action-policy tests. (1.0.0 E2 baseline: 172 passed, 2 skipped) | — |
| Smoke test | `make smoke` | E1 + E2 | PASS | all 4 stages green (`SMOKE TEST PASSED`) | — |
| Artifact checksum verification | `make verify` | E1 + E2 | PASS | 1.1.0: `236 artifact checks passed, 1 skipped, 0 failed` (manifests now also pin `action_policy.yaml`, the e7 groundedness CSVs, and the Table 3 reference); the 1 skip is the untracked compiled `paper/main.pdf` | adapter LFS objects must be fetched (`git lfs install`) or `--no-lfs` reports them SKIPPED |
| Manuscript tables regeneration (Tables 3–6, 8–11) | `make reproduce-paper` | E1 + E2 | PASS | all 8 generated `.tex` files **byte-identical** to `artifacts/reference/tables/` (Table 3 via `scripts/export_action_policy.py`) | Tables 1–2 have no numeric content; Table 7 is hand-typeset from a released, regenerable CSV (see below) |
| Manuscript figures regeneration (Figures 2–4) | `make reproduce-paper` | E1 + E2 | PASS | all 3 PNGs (300 dpi) **byte-identical**; PDFs identical modulo embedded timestamps | figure PNG byte-identity requires matplotlib 3.10.x (pinned); mpl 3.11 renders differently (caught by the comparison, verified) |
| Statistics regeneration (Table 9 inputs) | `make reproduce-stats` | E1 + E2 | PASS | `F1_physical.csv`, `F2_adversarial.csv`, `F3_gate.csv` regenerate **hash-identically** (fixed RNG 20260811, 20k permutations / 10k bootstrap; resampling draws shared across contrasts — protocol amended 2026-08-29, see CHANGELOG) | numpy/scipy major-version drift would be flagged by the digest comparison, not silently accepted |
| Latency summary derivation (Table 11) | `make derive-latency` | E1 + E2 | PASS | value-identical re-derivation from the 4 archived `predictions.jsonl` runs | the underlying latency measurements are E1 hardware-specific |
| Evidence-Gate one-shot evaluation | `make gate-evaluate` | E2 | PASS | regenerated `gate_v3_raw.csv` / `gate_v3_summary.csv` **byte-identical** to canonical (verified by `make verify` after re-run) | — |
| EH estimator robustness | `make eh-robustness` | E2 | PASS | regenerated outputs byte-identical to canonical | — |
| Deadline / SLO re-analysis | `make deadline` | E2 | PASS | regenerated outputs byte-identical to canonical | — |
| Command-authentication sub-study (Table 7 data) | `make gate-auth` | E2 | PASS | regenerated `gate_auth_{raw,summary}.csv` byte-identical to canonical | the Table 7 `.tex` itself is hand-typeset from this CSV (checksummed reference; no TeX generator existed for the submitted file) |
| Holdout, deterministic + oracle arms | `make holdout-cpu` | E2 | PASS | re-simulated D0/OPROJ from scratch; all 1,176 rows of the merged raw CSV **value-identical** to canonical (byte digest changes because the CSV is rewritten — documented in docs/EXPERIMENTS.md) | — |
| Structural enumeration + mutation guards | `make property-tests` | E2 | PASS | regenerated result JSON content-identical: 40,824 states / 0 violations; mutations caught with 1008 / 144 / 3930 violations | — |
| Ground-truth leakage / runtime-safe invariants | `pytest code/Multi_AI_Agent/test_runtime_safe_gate.py` | E1 + E2 | PASS | 8/8 including the flip-all-`tampered`-flags invariance test | OPROJ oracle arm is the single labelled exception (upper bound), by design |
| Strict LLM serving path (released adapter loads and serves) | one-call smoke with `DER_LLM_STRICT=1` | E1 | PASS (E1 only) | real call served: `backend=real_lora`, `substrate=cuda-4bit-nf4`, `adapter_sha=0c9dec242e2f` (matches every canonical run record) | needs GPU + base weights; not repeatable in E2 |
| Holdout LLM arms (Q1/QPROJ/L1/LPROJ/bareQ/bareL) | `make holdout-llm` | — | NOT RE-RUN | canonical raw rows shipped + checksummed; zero-fallback and metric invariants enforced by tests over the shipped data | full re-run needs GPU + base weights + hours; bitwise transcript identity across different GPU/CUDA stacks is not claimed |
| 70-case adversarial suite (both backbones) | `make adversarial` | — | NOT RE-RUN | canonical raw/summary rows + leakage report shipped + checksummed; suite regenerates deterministically (seed 0) from the released generator | same as above |
| OpenDSS power-flow check | `make opendss-check` | — | NOT RE-RUN | canonical raw/summary shipped + checksummed (12 configs, 0 non-converged) | needs `opendssdirect` install; not executed in E2 during release verification |
| QLoRA training | `code/finetuning/train.py` | — | NOT RE-RUN | released adapters carry full SHA-256 + resolved training configs (seed 0) | bit-identical re-training across hardware is not expected; released adapters are canonical |
| Gate calibration (frozen-point derivation) | `make gate-calibrate` | — | NOT RE-RUN | frozen JSON shipped, self-hash + manifest pins verified | re-calibration overwrites the frozen file (guarded by `make verify` flagging the change) |
| Supplementary-material tables/figures | legacy builders in `code/evaluation/` | — | NOT RE-RUN | canonical inputs (`lora_eval/…`) shipped + checksummed; builders ship | regeneration of supplementary items not re-verified for this release; main-text principal results were |
| Docker image | `docker build -t der-safeagent .` | — | NOT RUN | Dockerfile provided (python:3.12-slim + core requirements + verification ladder as CMD); contents mirror the E2 procedure that passed | Docker is not installed on the release-preparation machine, so the image build itself is unverified |
| CI workflow | `.github/workflows/ci.yml` | — | NOT RUN | workflow mirrors the E2 ladder exactly (checkout with LFS, install, compileall, pytest, smoke, verify, derive-latency, reproduce-paper) | cannot execute before the repository exists on GitHub |
| Manuscript build (tracked source, de-anonymized, URL-bearing availability) | `make latex` | E1 | PASS | `main.pdf` builds from `paper/` (52 pp, 2026-08-29); `Code and Data Availability` appears exactly once and cites the verified repository URL; no undefined refs/citations; no placeholder markers or machine-local absolute paths in body text | built with tectonic (XeTeX); page count grew from 49 to 52 with the author block, Table 3, and the Related-Work additions |
| Action-policy single source of truth | `make policy-export` + `pytest tests/test_action_policy.py` | E1 | PASS | YAML + Table 3 regenerate byte-identically from `safety_projection.py`; exhaustive sweep confirms a model-proposed irreversible primitive never executes in any exported class | policy is author-specified; conformance, not operational validation |
| Repository hygiene | `make hygiene` | E1 | PASS | 0 findings over 680+ files (secrets, stale anonymization wording, oversize non-LFS files, absolute paths outside the 7 documented frozen CSVs) | — |

## Known unresolved reproducibility limitations (honest list)

1. **LLM re-execution is protocol-reproducible, not bitwise-portable.**
   Greedy decoding is used throughout, but token-level identity across
   different GPU models/CUDA/driver/kernel-library versions has not been
   demonstrated and is not claimed.
2. **Base-model revisions were not pinned at training time**
   (`revision: null`). Compensating control: every serving trace records a
   hash of the loaded base `config.json`; drift is detectable, not
   preventable.
3. **Qwen adapter working-tree discrepancy (resolved in this release,
   disclosed here).** The internal tree held two near-identical Qwen
   adapter copies; the release ships the one matching the digest recorded
   in every run manifest (`0c9dec242e2f…`). See `docs/MODEL_PROVENANCE.md`.
4. **Two manuscript tables are not script-generated** (Table 2 descriptive;
   Table 7 hand-typeset from a released, byte-reproducible CSV). Shipped as
   checksummed references; see `docs/PAPER_ARTIFACT_MAP.md`.
5. **Per-run holdout traces (463 MB) are not in the git tree**; the
   aggregated raw CSV and the forensics computed from the traces are.
   Regenerating the traces requires re-running the holdout arms.
6. **Supplementary-material regeneration not re-verified** for this
   release; its canonical inputs are shipped and checksummed.
7. **Expert-evaluation study not executed** (human raters + IRB);
   recorded as such in `experiment_registry.json`.
8. **Docker build and GitHub Actions run unverified** on the preparation
   machine (no docker daemon; repository not yet on GitHub). Their steps
   are byte-for-byte the E2 procedure that passed locally.
