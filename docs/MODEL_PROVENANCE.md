# Model Provenance

## Base models (NOT redistributed)

| Backbone | Hugging Face ID | License | Access |
|---|---|---|---|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 | open download |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 Community License | gated — accept the license on Hugging Face and authenticate (`hf auth login`) |

Base weights must be obtained by the user under those licenses. The code
resolves base models via `code/llm_serving/model_paths.py`:

```bash
# default: the HF IDs above (normal transformers cache/download flow)
# or point at a local snapshot:
export DER_QWEN_BASE=/path/to/qwen2_5_7b
export DER_LLAMA_BASE=/path/to/llama3_1_8b
```

**Revision pinning limitation (stated honestly):** the adapters were trained
with `revision: null` (no base-model commit pin). Every serving trace
records a hash of the loaded base model's `config.json`
(`llm_model_config_hash`) as a compensating check, and the archived
`environment.json` records the experiment stack, but a byte-level base
snapshot pin does not exist. If upstream re-uploads ever change the base
weights, use the recorded config hash to detect the drift.

## QLoRA adapters (released, via Git LFS)

| Adapter | Path | SHA-256 (full) | Short (as recorded in run manifests) |
|---|---|---|---|
| Qwen2.5-7B | `code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter/adapter_model.safetensors` | `0c9dec242e2fd697c14fe49ecfcd57b00aea9b5d90de152b1638793afceac1e7` | `0c9dec242e2f` |
| Llama-3.1-8B | `code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter/adapter_model.safetensors` | `9a2b8436ce9c4bd2c078626c2435a8cf7b61884d7c7a1ad8b9f84b44d89128d7` | `9a2b8436ce9c` |

Every LLM-labelled result row / run manifest in the canonical results
records the short adapter SHA; the two digests above match those recordings
exactly (`make verify` re-checks the full digests).

**Packaging note.** In the internal working tree, the Qwen run directory
contained two copies of the adapter weights: `checkpoint-63/` (digest
`0c9dec242e2f…`, the copy every recorded run manifest points to) and
`adapter/` (digest `e48e5186…`, a later re-save differing in 8 of ~22.6 M
values of a single LoRA-B tensor, max |Δ| ≈ 1e-3). This release ships the
**provenance-matching** `0c9dec242e2f…` weights. The Llama adapter had a
single consistent copy.

`adapter_config.json` in the released adapters points
`base_model_name_or_path` at the HF IDs (the training machine's local path
was rewritten for portability; LoRA weights are byte-identical to the
recorded digests).

## Training configuration (recorded in `config_resolved.yaml` per run)

- LoRA r=16, alpha=32, dropout=0.05, target modules = all 7 projection
  matrices (q/k/v/o/gate/up/down)
- QLoRA: 4-bit NF4 base quantization, bf16 compute, gradient checkpointing
- max sequence length 512; 3 epochs; lr 2e-4 cosine; `paged_adamw_8bit`;
  batch 1 × grad-accum 8; **seed 0**
- Training data: the released 200-example synthetic corpus
  (`code/finetuning_dataset/processed/`, 161/21/18 split, per-row SHA-256 in
  `processed/manifest.json`)
- Recipes: `code/finetuning/configs/lora_qwen25_7b_local.yaml`,
  `lora_llama31_8b_local.yaml`; entry point `code/finetuning/train.py`

## Serving (inference-time) assertions

`code/llm_serving/local_lora.py` serves base+adapter locally (no vendor
APIs). Guarantees enforced in code and tests:

- **Strict mode** (`DER_LLM_STRICT=1`, set by every LLM-labelled driver): if
  the real model/adapter cannot serve a call, a hard `LLMBackendUnavailable`
  is raised — the heuristic fallback can never masquerade as model output.
  Canonical results contain zero fallback rows (asserted at run time and by
  `test_metric_invariants.py`).
- **Per-call provenance trace**: backend, base model, adapter path, adapter
  SHA (12-hex), base-config hash, substrate (`cuda-4bit-nf4` / `cuda-bf16` /
  `cpu-bf16`), prompt SHA-256, and any fallback reason.
- **Greedy decoding**: `temperature=0.0` → `do_sample=False`;
  `max_new_tokens=128` in the evaluation drivers. K=3 self-consistency =
  three deterministic prompt-strategy variants + majority vote.
- An adapter contributing zero LoRA tensors to the loaded model is rejected
  (guards against silently mismatched adapters).

## Hardware used for the canonical runs

RTX 3080 (10 GiB), CUDA 12.4, torch 2.6.0+cu124, transformers 4.57.6,
peft 0.14.0, bitsandbytes 0.49.2 — archived in
`code/results/ijcip_final_v3/environment.json`. One backbone is loaded at a
time (~10 GB VRAM with 4-bit NF4); a K=1 call is ≈5.5 s on this card
(Table 11).
