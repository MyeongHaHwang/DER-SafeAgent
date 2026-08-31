"""Evaluate a trained LoRA adapter on the held-out DER/ICS test split.

Computes:
    * exact-attack-class accuracy        (vs. ground-truth assistant turn)
    * recommended-action accuracy        (vs. ground-truth assistant turn)
    * JSON-validity rate
    * schema-compliance rate (required fields + enum membership)
    * mean / p95 per-sample latency in ms (K=1 greedy, K=3 self-consistency)

Outputs:
    code/results/ijcip_lora_eval/<adapter_tag>/eval_metrics.json
    code/results/ijcip_lora_eval/<adapter_tag>/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

import torch
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                             BitsAndBytesConfig)


REQUIRED_KEYS = (
    "attack_class", "affected_asset", "confidence",
    "rationale", "recommended_action", "expected_impact_tier",
)
ATTACK_CLASSES = {"none", "fdi", "replay", "command_spoof", "dos", "firmware"}
ACTIONS = {"no_op", "freeze_setpoint", "throttle_ramp",
           "request_ied_revalidation", "isolate_inverter"}
IMPACT_TIERS = {"negligible", "low", "medium", "high", "severe"}


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:
        return None


def _schema_ok(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    for k in REQUIRED_KEYS:
        if k not in obj:
            return False
    if obj["attack_class"] not in ATTACK_CLASSES:
        return False
    if obj["recommended_action"] not in ACTIONS:
        return False
    if obj["expected_impact_tier"] not in IMPACT_TIERS:
        return False
    try:
        c = float(obj["confidence"])
        if c < 0.0 or c > 1.0:
            return False
    except Exception:
        return False
    return True


def _load_model(base: str, adapter: str | None, four_bit: bool):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"device_map": "auto"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.no_grad()
def _generate_once(tok, model, messages: list[dict], max_new_tokens: int,
                     do_sample: bool, seed: int | None = None) -> tuple[str, float]:
    prompt_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt_text, return_tensors="pt").to(model.device)
    if seed is not None:
        torch.manual_seed(seed)
    t0 = time.perf_counter()
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.7 if do_sample else 1.0,
        top_p=0.9 if do_sample else 1.0,
        pad_token_id=tok.eos_token_id,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    completion = tok.decode(
        out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return completion, dt


def _self_consistency(preds: list[dict | None]) -> dict | None:
    cls_votes: dict[str, int] = {}
    for p in preds:
        if p and p.get("attack_class") in ATTACK_CLASSES:
            cls_votes[p["attack_class"]] = cls_votes.get(p["attack_class"], 0) + 1
    if not cls_votes:
        return next((p for p in preds if p is not None), None)
    winner = max(cls_votes, key=cls_votes.get)
    # take the highest-confidence prediction with the winning class
    cands = [p for p in preds if p and p.get("attack_class") == winner]
    cands.sort(key=lambda p: -float(p.get("confidence", 0.0)))
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base model dir")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapter dir")
    ap.add_argument("--test", required=True, help="test.jsonl path")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--tag", required=True, help="adapter tag (qwen2_5_7b / llama3_1_8b / base / heuristic)")
    ap.add_argument("--four-bit", action="store_true", default=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    out_dir = Path(args.out) / args.tag; out_dir.mkdir(parents=True, exist_ok=True)

    tok, model = _load_model(args.base, args.adapter, args.four_bit)

    n = 0; n_json = 0; n_schema = 0; n_class = 0; n_action = 0
    lat1 = []; latK = []
    preds_log: list[dict] = []

    with open(args.test) as fh:
        for line in fh:
            row = json.loads(line)
            # ground truth is the final assistant message in the chat
            gt_msg = row["messages"][-1]
            assert gt_msg["role"] == "assistant"
            gt = _extract_json(gt_msg["content"]) or {}
            prompt_msgs = row["messages"][:-1]
            # K=1 greedy
            text1, dt1 = _generate_once(tok, model, prompt_msgs,
                                            args.max_new_tokens, do_sample=False)
            lat1.append(dt1)
            pred1 = _extract_json(text1)
            # K-sample self-consistency
            samples: list[dict | None] = [pred1]
            dt_extra = 0.0
            for seed in range(1, args.k):
                ti, di = _generate_once(tok, model, prompt_msgs,
                                          args.max_new_tokens, do_sample=True,
                                          seed=seed)
                samples.append(_extract_json(ti))
                dt_extra += di
            latK.append(dt1 + dt_extra)
            pred = _self_consistency(samples) or pred1 or {}

            n += 1
            valid = pred is not None and isinstance(pred, dict) and bool(pred)
            n_json += 1 if valid else 0
            sok = _schema_ok(pred)
            n_schema += 1 if sok else 0
            if sok and gt.get("attack_class") == pred.get("attack_class"):
                n_class += 1
            if sok and gt.get("recommended_action") == pred.get("recommended_action"):
                n_action += 1
            preds_log.append({
                "id": row.get("id"),
                "true_attack_class": gt.get("attack_class"),
                "true_recommended_action": gt.get("recommended_action"),
                "pred_attack_class": pred.get("attack_class") if isinstance(pred, dict) else None,
                "pred_recommended_action": pred.get("recommended_action") if isinstance(pred, dict) else None,
                "json_valid": valid, "schema_ok": sok,
                "latency_k1_ms": dt1, "latency_kK_ms": dt1 + dt_extra,
            })

    def _pct(num, denom):
        return (num / denom) if denom else float("nan")

    def _p95(values):
        if not values: return float("nan")
        s = sorted(values)
        return s[int(len(s) * 0.95) - 1] if len(s) >= 20 else s[-1]

    metrics = {
        "tag": args.tag,
        "base": args.base,
        "adapter": args.adapter,
        "k": args.k,
        "n_samples": n,
        "json_validity_rate": _pct(n_json, n),
        "schema_compliance_rate": _pct(n_schema, n),
        "attack_class_accuracy": _pct(n_class, n),
        "action_accuracy": _pct(n_action, n),
        "latency_k1_ms_mean": mean(lat1) if lat1 else float("nan"),
        "latency_k1_ms_p95": _p95(lat1),
        "latency_kK_ms_mean": mean(latK) if latK else float("nan"),
        "latency_kK_ms_p95": _p95(latK),
    }
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.jsonl").open("w") as fh:
        for r in preds_log:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
