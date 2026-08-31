# DER-SafeAgent

![DER-SafeAgent](./DER-SafeAgent_Logo.png)

**A Runtime-Assurance Architecture for Safe LLM-Assisted Cyber-Physical
Incident Response in Distributed Energy Resources.**

Reproducibility package for the manuscript submitted to the *International
Journal of Critical Infrastructure Protection* (IJCIP). It contains the
implementation, the frozen experiment protocols, the synthetic datasets, the
released QLoRA adapters, the canonical result files, the manuscript LaTeX
source, and the scripts that regenerate every numeric table and figure in
the manuscript. A claim-by-claim audit connecting each reported number to
its raw source is kept in [`docs/CLAIM_LEDGER.md`](docs/CLAIM_LEDGER.md).

> **What this system claims — and what it does not.** DER-SafeAgent treats a
> locally served LLM (Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct with
> QLoRA adapters) as an **untrusted advisory component** inside a
> deterministic runtime-assurance boundary: an observable-evidence gate, a
> safety-projection shield over a fixed five-action mitigation registry,
> mandatory human escalation of irreversible actions, a symmetric veto rule
> (model output may gate neither irreversible action nor stand-down
> inaction), a deterministic fast path that survives model failure, and a
> hash-chained audit record. **The containment guarantees come from the
> deterministic mechanisms, not the model**; the paper does not claim the
> LLM improves physical outcomes. See `docs/THREAT_MODEL.md`.

> **Naming.** The publication-facing system name is **DER-SafeAgent**. The
> Python package (`code/`, `Multi_AI_Agent`, …), module paths, and result
> directories retain the legacy name **DER-SecAgent**: renaming them would
> invalidate the configuration hashes recorded in the frozen manifests.
> Wherever the paper says *DER-SafeAgent*, the code says `DER-SecAgent`.

## Repository structure

```
├── code/                    # research implementation (legacy package name)
│   ├── Multi_AI_Agent/      #   runtime path: FeatureView, Evidence Gate, shield,
│   │                        #   estimators, registry, audit chain, prompts + safety tests
│   ├── simulation/          #   StubFeeder/OpenDSS harness, attack injectors, scenarios
│   ├── llm_serving/         #   strict local QLoRA serving (no vendor APIs)
│   ├── baselines/           #   deterministic policy, single-LLM baseline, detectors
│   ├── evaluation/          #   experiment drivers; final_safeagent_v3/ = canonical pipeline
│   ├── finetuning/          #   QLoRA recipes + released adapters (Git LFS)
│   ├── finetuning_dataset/  #   synthetic 200-example corpus + builders + data card
│   ├── configs/             #   hash-frozen protocol manifests (byte-exact)
│   ├── datasets/            #   external-benchmark loaders + mock fixture (no 3rd-party data)
│   ├── results/             #   canonical raw/processed results (~2 MB, checksummed)
│   └── docs/                #   threat model, AI-pipeline taxonomy
├── artifacts/               # reference (manuscript-exact) + regenerated tables/figures
├── data/manifests/          # SHA-256 release manifests
├── docs/                    # REPRODUCIBILITY, EXPERIMENTS, provenance, artifact map, CLAIM_LEDGER
├── paper/                   # manuscript LaTeX source (build: tectonic main.tex)
├── scripts/                 # bootstrap, smoke, verify, reproduce, policy export, environment
├── tests/                   # repo-level tests (manifests, artifacts, action policy, smoke)
├── Makefile                 # all entry points (`make help`)
└── REPRODUCIBILITY_REPORT.md / RELEASE_CHECKLIST.md / RELEASE_AUDIT.md
```

## Installation

Python ≥ 3.11 (verified on 3.12.3, Linux). From the repository root:

```bash
git lfs install                      # adapters are stored via Git LFS
bash scripts/bootstrap.sh            # venv + core deps + smoke test + verification
# or manually:
pip install -r requirements.txt      # CPU verification stack
```

`requirements-lock.txt` pins the exact environment in which this release was
verified; `environment.yml` (conda) and `Dockerfile` (CPU image) are
provided. GPU stack for full re-execution: `make setup-llm`.

## Five-minute smoke test (CPU)

```bash
make test     # full unit + safety-invariant suite (179 tests, ~10 s)
make smoke    # imports, frozen configs, gate/shield invariants, mini episode
make verify   # SHA-256 of all released artifacts (236 checks)
```

The single all-in-one gate (tests, smoke, checksums, paper-artifact
regeneration, hygiene scan, and — when a TeX engine is available — the
manuscript build) is:

```bash
make release-check
```

## Regenerate the paper's tables and figures (CPU, no LLM)

```bash
make reproduce-paper
```

Regenerates Table 3 (action policy), Tables 4–6, and Tables 8–11 plus
Figures 2–4 from the canonical raw result files and compares against
`artifacts/reference/` (byte-exact copies of the manuscript's files).
Verified for this release: **all eight generated tables byte-identical; all
three figure PNGs byte-identical; statistics regenerate hash-identically**
(fixed RNG seed 20260811). Tables 1–2 and Figure 1 have no numeric content;
Table 7's underlying CSV regenerates with `make gate-auth`. Full mapping:
`docs/PAPER_ARTIFACT_MAP.md`.

## Full model-in-the-loop experiments (GPU)

```bash
make setup-llm                        # torch/transformers/peft/accelerate/bitsandbytes
# obtain base weights under their licenses (docs/MODEL_PROVENANCE.md):
#   Qwen/Qwen2.5-7B-Instruct (Apache-2.0), meta-llama/Llama-3.1-8B-Instruct (gated)
make holdout-cpu holdout-llm          # 8 arms x 49 frozen configs x 3 episodes
make adversarial                      # 70-case suite, both backbones
```

Requirements: NVIDIA GPU with ≥ 10 GB VRAM (4-bit NF4, one backbone at a
time); reference: RTX 3080, CUDA 12.4, ≈5.5 s per K=1 call, multiple hours
per backbone. Every LLM stage runs under **strict serving**
(`DER_LLM_STRICT=1`): a serving failure is a hard error, and heuristic
fallbacks can never be recorded as model output (zero fallback rows in all
canonical results, enforced by tests). Stage-by-stage guide:
`docs/EXPERIMENTS.md`.

## Datasets and models

- **Released:** the fully synthetic 200-example fine-tuning corpus
  (CC0-1.0, per-row SHA-256), the 146-prompt eval corpus, all frozen
  protocol manifests, all canonical results, and both QLoRA adapters
  (Git LFS; Qwen `0c9dec242e2f…`, Llama `9a2b8436ce9c…` — full digests in
  `docs/MODEL_PROVENANCE.md`).
- **Not redistributed:** Qwen/Llama base weights (obtain under their own
  licenses), UNSW-MG24 and TON_IoT (research-data terms; loaders, a mock
  fixture, and acquisition notes ship instead — `data/README.md`).
- All DER telemetry and attack scenarios are synthetic or
  simulation-derived; no personal data or operational utility credentials
  exist anywhere in this repository.

## Verifying integrity

```bash
make verify                                   # everything (non-zero exit on failure)
python3 scripts/verify_artifacts.py --strict  # + flag unexpected files in frozen dirs
python3 scripts/derive_latency_v3.py          # Table 11 re-derivation check
make environment                              # record your environment for comparison
```

## Reproducibility scope and known limitations

Verified: CPU regeneration of all main-text numeric artifacts
(byte-identical), checksum integrity of all 236 released artifact checks, the
full test suite, and value-identical re-derivation of the latency summary.
Not guaranteed: bit-identical LLM outputs across GPU/CUDA/driver stacks
(greedy decoding is used, but cross-platform bitwise identity has not been
demonstrated and is not claimed); base-model revisions were not pinned at
training time (compensating config-hash check recorded per call);
supplementary-material regeneration was not re-verified for this release.
Full discussion and the complete verification record:
`docs/REPRODUCIBILITY.md` and `REPRODUCIBILITY_REPORT.md`.

## Citation

See [`CITATION.cff`](./CITATION.cff). The manuscript (Hwang, Kim, Kim,
Kwon, Lee) is under review at IJCIP; the archival reference will be added
on publication.

## License

Apache-2.0 for the project's own code, configurations, synthetic data, and
results — see [`LICENSE`](./LICENSE) for the explicit scope note covering
third-party models and datasets that are referenced but not distributed
(the Llama QLoRA adapter is additionally subject to the Llama 3.1
Community License; the synthetic corpus is CC0-1.0).

## Contact

Maintainer: Myeong-Ha Hwang (KEPRI) — <raphael9290@gmail.com>. Please use
GitHub issues for reproduction problems; see `SECURITY.md` for reporting
safety-relevant defects.
