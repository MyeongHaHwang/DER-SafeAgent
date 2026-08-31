# Reproducibility Protocol

This project commits to **bit-stable rerun** of every reported number from the
artifacts in this repo plus pinned upstream sources. Every result row in the
paper traces back to a `manifest.json` somewhere in `results/`.

## Pinned environment
- Python 3.11
- See `requirements.txt` (top-level) and `code/finetuning/requirements.txt`
  (training stage). Versions are upper-bounded so a `pip install` years from
  now still resolves a compatible set.

## Pinned models
| Role | Model ID | Notes |
|------|----------|-------|
| Base for fine-tuning (default) | `Qwen/Qwen2.5-7B-Instruct` | open weights |
| Base for fine-tuning (alt) | `meta-llama/Meta-Llama-3.1-8B-Instruct` | open weights, license-gated |
| Single-LLM baseline | `claude-haiku-4-5-20251001` | dated snapshot |
| Multi-agent system LLM | `claude-haiku-4-5-20251001` | same model, with tool use |

The exact ID is recorded in each run's `config_resolved.yaml` (training) and
`manifest.json` (evaluation).

## Pinned prompts
Every prompt template is a file in the repo. The SHA-256 (first 12 chars) is
hashed at run time and emitted into the run manifest. If the file changes, the
hash changes, and old results are tagged stale automatically.

| Component | Template | Hash recorded in |
|-----------|----------|------------------|
| Single-LLM baseline | `code/baselines/single_llm/prompt.txt` | `manifest.json` |
| DER-SecAgent agents | `code/Multi_AI_Agent/prompts/*.txt` | per-agent block in manifest |

## Pinned dataset
- `code/finetuning_dataset/processed/manifest.json` lists every example
  hash and split assignment, scoped to a numbered version (v0.1, v1.0, ...).
- `build_dataset.py --seed 0` reproduces this manifest.
- Public-log subsets that cannot be redistributed are referenced by URL +
  expected SHA-256 in `code/finetuning_dataset/raw/SOURCES.md` (to be
  written before v1.0).

## Pinned random seeds
- Default seed list: `0,1,2,3,4`.
- Bootstrap: `seed=0`, 10,000 resamples.
- Stratified split: `--seed 0`.

## End-to-end reproduce
```bash
# 1. build dataset
cd code/finetuning_dataset && python build_dataset.py --seed 0 && python validate.py

# 2. fine-tune (optional — required only if reporting the fine-tuned variant)
cd ../finetuning && python train.py --config configs/lora_qwen25_7b.yaml

# 3. run full experimental matrix
cd ../.. && python -m code.evaluation.run_all --seeds 0,1,2,3,4

# 4. render figures
python -m code.evaluation.plot_cost_curves --input results/cost_curves.csv \
       --out paper/figure/cost_curves.pdf
```

## Provenance manifest
Each run directory contains:
- `manifest.json` — scenario, detector, seed, config hash, completion time
- `config_resolved.yaml` (training only) — fully resolved YAML
- `git_sha.txt` — repo SHA at run time
- `data_manifest.json` (training only) — hash-stable dataset version

Together these are sufficient to re-derive the outputs from sources.

## Things that are *not* pinned (stated for honesty)
- LLM API responses for `single_llm` and `prior_mas` are non-deterministic at
  temperature > 0. We run with deterministic decoding where supported, but
  vendor-side updates can change behavior between runs. We mitigate by
  reporting **5-seed mean ± bootstrap CI** and re-running on a frozen date
  before camera-ready.
- GPU non-determinism in fine-tuning: `transformers` does not guarantee
  bit-exact reruns across different CUDA versions. We report the GPU model
  and CUDA version in `manifest.json` for the training stage.
