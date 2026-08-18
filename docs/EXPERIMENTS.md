# Running the Experiments

This is the Mode-3 (`full-experiment`) companion to
`docs/REPRODUCIBILITY.md`. Everything here re-executes experiments; nothing
here is needed to verify the manuscript's tables/figures (that is
`make reproduce-paper`, CPU-only).

All commands run **from the repository root**. Drivers are module entry
points (`python3 -m code.…`) with repository-root-relative paths.

## Hardware / software requirements

| Stage | Hardware | Extra deps | Verified runtime (reference) |
|---|---|---|---|
| Gate calibrate/evaluate, gate-auth, EH robustness, deadline, holdout D0/OPROJ | any CPU | core `requirements.txt` | minutes each |
| OpenDSS check | any CPU | `make setup-dss` | minutes |
| Holdout LLM arms (Q1, QPROJ, L1, LPROJ, bareQ, bareL) | NVIDIA GPU, ≥10 GB VRAM (4-bit NF4), one backbone at a time | `make setup-llm` + base weights | ≈8 s/call K=1 on an RTX 3080; six arms × 49 configs × 3 episodes → multiple hours per backbone |
| 70-case adversarial suite (both backbones) | same GPU stack | same | hours (8 arms × 70 cases × sustained ticks, per backbone) |
| QLoRA re-training (optional) | same GPU stack | `code/finetuning/requirements.txt` | ~1 h/backbone on RTX 3080 (63 steps) |

Reference environment for all canonical GPU results:
`code/results/ijcip_final_v3/environment.json` (RTX 3080 10 GiB, CUDA 12.4,
torch 2.6.0+cu124, transformers 4.57.6, peft 0.14.0, bitsandbytes 0.49.2).

## Base-model acquisition

See `docs/MODEL_PROVENANCE.md`. Short version:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login                      # needed for the gated Llama repo
# default flow downloads on first use via the HF IDs; or pre-download:
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/qwen2_5_7b
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama3_1_8b
export DER_QWEN_BASE=$PWD/models/qwen2_5_7b
export DER_LLAMA_BASE=$PWD/models/llama3_1_8b
```

The QLoRA adapters are already in the repository (Git LFS): verify with
`make verify` before use.

## Strict serving — the integrity switch

Every LLM-labelled stage exports `DER_LLM_STRICT=1` (the Makefile does this
for you). Under strict mode a serving failure raises immediately; **no
heuristic-fallback output can ever be recorded as model output**. The
drivers additionally hard-assert `n_fallbacks == 0` per run. If a stage
aborts with `LLMBackendUnavailable` / `FATAL: … backend unavailable`, fix
the model setup — do not remove the flag.

## Stage-by-stage

### Evidence Gate (CPU)

```bash
make gate-evaluate        # one-shot eval on the frozen 32-config test set
                          #   -> code/results/ijcip_final_v3/gate_robustness/
make gate-calibrate       # OPTIONAL: re-derive the frozen operating point from dev data.
                          # WARNING: overwrites code/configs/ijcip_final_v3/evidence_gate_v3_frozen.json;
                          # `make verify` will then flag the digest change, by design.
make gate-auth            #   -> code/results/ijcip_final/gate_auth/
```

### Estimator + latency re-analyses (CPU)

```bash
make eh-robustness        # 12-config holdout x 10 stress conditions
make deadline             # SLO admission from archived strict-serving predictions
make derive-latency       # verifies Table 10's summary re-derives value-identically
```

### End-to-end holdout (CPU arms, then GPU arms)

```bash
make holdout-cpu          # D0 (deterministic fast path) + OPROJ (oracle bound)
make holdout-llm          # Q1,QPROJ then L1,LPROJ then bareQ,bareL (GPU)
```

Notes: results append/dedupe into
`code/results/ijcip_final_v3/holdout_e2e/holdout_e2e_raw.csv`; per-run
traces (decisions.jsonl, timeseries.csv, manifest.json) are written under
`holdout_e2e/runs/` (git-ignored). A run whose `manifest.json` exists is
skipped, so interrupted arms resume. `make holdout-freeze` regenerates the
scenario set; it must reproduce the frozen manifest digest
(`df918691…` — `make verify` checks).

Re-running any holdout arm rewrites `holdout_e2e_raw.csv`, so its byte
digest changes and `make verify` flags it — by design. Verified during
release preparation: re-running `make holdout-cpu` on a fresh CPU-only
environment reproduced all D0/OPROJ rows **value-identically** (1,176 rows
compared after sorting); use a DataFrame comparison, not a byte comparison,
after intentional re-runs.

### 70-case adversarial suite (GPU)

```bash
make adversarial          # qwen backend, then llama backend
```

Includes the run-time leakage check against the fine-tuning corpus
(`leakage_report.csv` / `leakage_summary.json` must report zero duplicates).

### OpenDSS power-flow check (CPU)

```bash
make setup-dss && make opendss-check    # -> code/results/ijcip_final/opendss_check/
```

### Statistics + artifacts after re-running

```bash
make reproduce-stats      # permutation/bootstrap/Holm from the (new) raw CSV
SOURCE_DATE_EPOCH=0 python3 -m code.evaluation.final_safeagent_v3.build_artifacts_v3
python3 -m pytest code/evaluation/final_safeagent_v3/test_metric_invariants.py -q
```

If you re-ran with the archived protocol on the archived stack, compare with
`python3 scripts/compare_artifacts.py`. On different hardware, expect
protocol-level agreement (containment counts, coverage), not bitwise
transcript equality — see the determinism section of
`docs/REPRODUCIBILITY.md`.

### QLoRA re-training (optional)

```bash
pip install -r code/finetuning/requirements.txt
cd code/finetuning && python train.py --config configs/lora_qwen25_7b_local.yaml
```

Training is seeded (seed 0) but GPU kernels make bit-identical adapters
across hardware unlikely; the released adapters are the canonical ones used
in the paper.

## Output locations (summary)

| Stage | Output |
|---|---|
| gate evaluate / calibrate | `code/results/ijcip_final_v3/gate_robustness/`, frozen JSON in `code/configs/ijcip_final_v3/` |
| gate-auth | `code/results/ijcip_final/gate_auth/` |
| EH robustness | `code/results/ijcip_final_v3/eh_robustness/` |
| deadline / SLO | `code/results/ijcip_final_v3/deadline/` |
| holdout (all arms) | `code/results/ijcip_final_v3/holdout_e2e/` |
| adversarial | `code/results/ijcip_final_v3/adversarial/` |
| OpenDSS check | `code/results/ijcip_final/opendss_check/` |
| statistics | `code/results/ijcip_final_v3/statistics/` |
| tables/figures | `artifacts/tables/`, `artifacts/figures/` |
