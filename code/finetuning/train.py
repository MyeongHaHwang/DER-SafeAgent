"""LoRA SFT trainer for DER-SecAgent-LM.

Reads a YAML config (see configs/), loads the JSONL splits emitted by
`code/finetuning_dataset/build_dataset.py`, and trains a LoRA adapter.
Assistant-only loss masking is applied so we don't train on the user/system tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ASSISTANT_TAG = "<|im_start|>assistant"  # ChatML; override per tokenizer if needed


def _assistant_marker_for(tokenizer) -> str:
    """Best-effort marker for the start of the last assistant turn so we can
    mask the user/system prefix from the loss. Returns a string that is safe
    to tokenise with ``add_special_tokens=False``; if the tokenizer's chat
    template does not expose the ChatML start-tag we fall back to the bare
    word ``assistant`` (still produces a useful cut for most templates)."""
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    if "<|im_start|>" in vocab:
        return "<|im_start|>assistant"
    if "<|start_header_id|>" in vocab:
        return "<|start_header_id|>assistant<|end_header_id|>"
    return "assistant"


def render_chatml(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_dataset(tokenizer, jsonl_path: str, max_len: int):
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    marker_text = _assistant_marker_for(tokenizer)
    marker_ids = tokenizer(marker_text, add_special_tokens=False)["input_ids"]

    def encode(row):
        text = render_chatml(tokenizer, row["messages"])
        ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        labels = list(ids)
        # mask everything before the *last* assistant turn — train only on that
        cut = _find_last_subseq(ids, marker_ids)
        for i in range(min(cut, len(labels))):
            labels[i] = -100
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    return ds.map(encode, remove_columns=ds.column_names)


def _find_last_subseq(seq: list[int], sub: list[int]) -> int:
    if not sub:
        return 0
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i + len(sub)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + Path(args.config).stem
    out_dir = Path("results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = (cfg.get("quantization") or "").lower()
    if quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], quantization_config=bnb, device_map="auto",
        )
        # Minimal kbit prep without the fp32 norm/embed upcast that pushes a
        # 7-8B base + LoRA over the 10 GB envelope; norms/embeds stay in
        # bf16, which is fine for short-context LoRA SFT on this corpus.
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if cfg["training"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], torch_dtype=torch.bfloat16, device_map="auto",
        )
    model = get_peft_model(model, LoraConfig(**cfg["lora"]))
    model.print_trainable_parameters()

    train_ds = build_dataset(tokenizer, cfg["data"]["train"], cfg["max_len"])
    eval_ds = build_dataset(tokenizer, cfg["data"]["val"], cfg["max_len"])

    targs = TrainingArguments(
        output_dir=str(out_dir),
        report_to="none",
        bf16=cfg.get("bf16", True),
        seed=cfg["seed"],
        **cfg["training"],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))

    # provenance
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except FileNotFoundError:
        git_sha = ""
    (out_dir / "git_sha.txt").write_text(git_sha or "unknown")
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg))
    manifest_src = Path(cfg["data"]["train"]).parent / "manifest.json"
    if manifest_src.exists():
        (out_dir / "data_manifest.json").write_text(manifest_src.read_text())
    print(json.dumps({"run_id": run_id, "out": str(out_dir)}))


if __name__ == "__main__":
    main()
