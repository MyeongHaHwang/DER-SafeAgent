# Reproducibility Guide

This document defines exactly what can be reproduced, how, on what hardware,
and with what determinism guarantees. The honest one-line summary:

> **Every numeric table and figure in the manuscript regenerates from the
> released canonical result files on a CPU-only machine, byte-identically
> (tables, figure PNGs) — verified. Re-running the LLM-in-the-loop
> experiments themselves requires a GPU and the base-model weights, and is
> reproducible in protocol and provenance, but not guaranteed bit-identical
> across GPU/driver/library versions.**

## Three reproduction modes

### Mode 1 — `smoke` (minutes, CPU, no model)

```bash
make setup      # pip install -r requirements.txt
make smoke
```

Checks imports, configuration loading (frozen gate operating point, 49-config
holdout manifest), schema validation of the fine-tuning corpus, the
runtime-safe FeatureView, the Evidence Gate, the safety-projection shield,
the fixed action registry, audit-chain behaviour, and a miniature stochastic
episode. Uses small fixtures only; no GPU, no LLM, no OpenDSS.
Verified runtime: well under 1 minute after installation.

### Mode 2 — `reproduce-paper` (minutes, CPU, no model)

```bash
make verify            # SHA-256 of all 200+ released artifacts
make reproduce-paper   # stats -> tables/figures -> comparison
```

Regenerates the manuscript's numeric artifacts from the canonical raw result
files (~300 KB of CSVs) — Tables 3–6 and 8–11 and Figures 2–4 — and compares
them against `artifacts/reference/`. This mode does **not** re-run any LLM
inference; it proves the mapping raw results → published numbers.
Verified: all 7 generated tables byte-identical; all 3 figure PNGs
byte-identical; the 3 statistics CSVs (permutation/bootstrap with fixed RNG
seed 20260811) regenerate hash-identically. Figure PDFs match modulo embedded
timestamp metadata (PDF stream packing is matplotlib-version-dependent; the
300-dpi PNG is the strict pixel check).

### Mode 3 — `full-experiment` (hours–days, GPU)

Re-runs the actual experiments. CPU stages first:

```bash
make gate-evaluate     # one-shot Evidence-Gate eval on the frozen 32-config set
make gate-auth         # command-authentication sub-study (seeds 2101-2103)
make eh-robustness     # EH estimator, 12-config holdout x 10 conditions
make deadline          # SLO re-analysis of archived strict-serving predictions
make holdout-cpu       # deterministic (D0) + oracle (OPROJ) holdout arms
make opendss-check     # needs `make setup-dss` (opendssdirect)
```

GPU stages (see `docs/EXPERIMENTS.md` for model acquisition and runtimes):

```bash
make setup-llm
make holdout-llm       # Q1,QPROJ / L1,LPROJ / bareQ,bareL — strict serving
make adversarial       # 70-case suite, both backbones
make reproduce-stats   # then rebuild statistics from the new raw file
```

## Determinism and seeds — what is and is not guaranteed

Controlled and recorded:

- **Python/NumPy RNG seeds** for every stochastic component: episode seeds
  1301/1302/1303 (holdout, frozen in `holdout_v3_freeze.json`), suite seed 0
  with 5 cases/family (adversarial), calibration episode seeds (500+i),
  gate-auth seeds 2101–2103, statistics RNG 20260811 with 20,000
  permutations / 10,000 bootstrap resamples, corpus build seed 0,
  fine-tuning seed 0.
- **Frozen protocol hashes**: every dev/test/holdout manifest is SHA-256
  pinned inside its `*_freeze.json`; per-scenario `config_sha256` is recorded
  per run; `make verify` re-checks all pins.
- **Prompt and adapter provenance**: every LLM call trace records the prompt
  SHA-256, adapter SHA-256 (12-hex), base-model config hash, and the serving
  substrate (`cuda-4bit-nf4` / `cuda-bf16` / `cpu-bf16`).
- **Greedy decoding** (`temperature=0.0` → `do_sample=False`,
  `max_new_tokens=128` in the evaluation drivers); K=3 self-consistency runs
  three deterministic prompt-strategy variants (each greedy) and
  majority-votes — no sampling anywhere in the canonical runs.
- **Strict serving**: `DER_LLM_STRICT=1` makes any backend failure a hard
  error; the canonical results contain **zero** heuristic-fallback rows
  (enforced by `test_metric_invariants.py::test_no_serving_fallback…` and by
  runtime asserts in the drivers).

NOT guaranteed:

- **Bit-identical LLM outputs across hardware/software stacks.** The
  archived runs used an RTX 3080 (10 GB), CUDA 12.4, torch 2.6.0+cu124,
  transformers 4.57.6, peft 0.14.0, bitsandbytes 0.49.2 (4-bit NF4) — see
  `code/results/ijcip_final_v3/environment.json`. Greedy decoding on a
  different GPU model, CUDA version, driver, or kernel-library version can
  yield different token sequences on some inputs. We have not demonstrated
  bitwise identity across platforms and do not claim it. Protocol-level
  conclusions (containment counts, gate coverage) are the reproduction
  target for Mode 3, not bitwise transcript equality.
- **Statistics under different numpy/scipy majors.** The lock file
  (`requirements-lock.txt`) records the verified versions; `make
  reproduce-paper` fails loudly (checksum comparison) if version drift
  changes any statistic, rather than silently accepting it.

## Environment capture

```bash
make environment   # writes artifacts/environment_local.json
```

Compare with the archived `code/results/ijcip_final_v3/environment.json`
when investigating discrepancies.

## Verified-command matrix

The complete verification record for this release, with per-command status
and evidence, is in [`REPRODUCIBILITY_REPORT.md`](../REPRODUCIBILITY_REPORT.md).

## What is missing for full re-execution, honestly

1. **Base-model weights** are not redistributed (licenses). Obtain
   `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) and
   `meta-llama/Llama-3.1-8B-Instruct` (gated; Llama 3.1 Community License)
   yourself — `docs/MODEL_PROVENANCE.md`.
2. **Base-model revisions were not pinned** at training time
   (`revision: null` in the adapter configs). The adapters record the base
   config hash at load time, and `docs/MODEL_PROVENANCE.md` documents the
   snapshot dates, but an upstream re-upload of the base weights could in
   principle change behaviour. This is a known limitation, stated as such.
3. **Per-run trace dumps** (463 MB `holdout_e2e/runs/`) are not in the git
   tree; the raw per-configuration metrics CSV they aggregate into is. The
   two gate-invariance forensics JSONs computed from those traces are
   shipped with checksums; re-deriving them requires re-running the holdout.
4. **The expert-evaluation study** described in the registry was not
   executed (needs human raters + IRB); the registry records it as such.
5. **Supplementary-material regeneration** was not re-verified for this
   release (main-text principal results were; see `PAPER_ARTIFACT_MAP.md`).
