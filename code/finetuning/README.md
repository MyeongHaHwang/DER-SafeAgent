# Fine-tuning

Goal: produce a DER-security-specialized LLM ("DER-SecAgent-LM") that improves
detection F1 **and** energy-impact-aware action selection over the base model.

## Strategy
- **Method**: Supervised Fine-Tuning (SFT) with **LoRA** adapters (r=16, α=32).
- **Backbones (compared)**:
  - Qwen2.5-7B-Instruct (default)
  - Llama-3.1-8B-Instruct
- **Data**: `../finetuning_dataset/processed/{train,val}.jsonl` (ChatML).
- **Loss**: standard causal LM with assistant-only token masking.
- **DPO**: deferred to v2; requires preference pairs we don't yet have.

## Files
- `train.py` — HF Trainer + PEFT, reads YAML config.
- `eval.py` — runs the held-out test split, emits `results/eval_<run_id>.json`.
- `configs/lora_qwen25_7b.yaml` — recipe for Qwen2.5-7B.
- `configs/lora_llama31_8b.yaml` — recipe for Llama-3.1-8B.
- `requirements.txt` — pinned deps for this stage.

## Reproducibility
Each run writes `results/<run_id>/{config_resolved.yaml,git_sha.txt,manifest.json}`
plus the LoRA weights. The `manifest.json` mirrors the dataset version hash from
`processed/manifest.json` so we can prove which rows trained which adapter.

## How to run
```bash
cd code/finetuning
pip install -r requirements.txt
python train.py --config configs/lora_qwen25_7b.yaml
python eval.py  --run-id <run_id_printed_by_train>
```
