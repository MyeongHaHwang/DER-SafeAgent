"""Evaluate a fine-tuned LoRA adapter on the held-out test split.

Reports:
- Detection metrics: per-class precision / recall / F1, macro-F1.
- Action quality: exact-match on `recommended_action` (placeholder until WS2 binds
  actions to actual mitigation primitives).
- Impact-tier accuracy.

Energy-impact metrics (ENS, voltage, ramp) are *not* computed here — they require
the co-sim harness from WS2 (`code/simulation/harness.py`). After that exists,
call `evaluation.run_all` instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_assistant(text: str) -> dict:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    run_dir = Path(args.results_dir) / args.run_id
    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())

    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(run_dir / "adapter"))
    model.eval()

    ds = load_dataset("json", data_files=cfg["data"]["test"], split="train")

    y_true_class: list[str] = []
    y_pred_class: list[str] = []
    y_true_tier: list[str] = []
    y_pred_tier: list[str] = []
    action_match = 0

    for row in ds:
        prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        gen = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

        pred = parse_assistant(gen)
        gold = parse_assistant(row["messages"][-1]["content"])
        y_true_class.append(gold.get("attack_class", "?"))
        y_pred_class.append(pred.get("attack_class", "?"))
        y_true_tier.append(gold.get("expected_impact_tier", "?"))
        y_pred_tier.append(pred.get("expected_impact_tier", "?"))
        if pred.get("recommended_action") == gold.get("recommended_action"):
            action_match += 1

    report = {
        "n": len(ds),
        "attack_class_report": classification_report(y_true_class, y_pred_class, output_dict=True, zero_division=0),
        "impact_tier_report": classification_report(y_true_tier, y_pred_tier, output_dict=True, zero_division=0),
        "action_exact_match": action_match / max(len(ds), 1),
    }
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"run_id": args.run_id, "n": report["n"],
                      "macro_f1_class": report["attack_class_report"]["macro avg"]["f1-score"]}))


if __name__ == "__main__":
    main()
