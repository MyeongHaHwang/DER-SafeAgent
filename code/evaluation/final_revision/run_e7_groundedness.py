"""E7: automated explanation-groundedness and audit validation.

Explanations are not evaluated by fluency. We check, from the raw model
outputs already stored with their prompts, whether the rationale is *grounded*
in the input it was given:

  * asset groundedness   - does the cited asset appear in the prompt?
  * class consistency    - does the rationale name the class the model predicted?
  * action consistency   - does the rationale name the action the model proposed?
  * unsupported claims   - does the rationale assert evidence (tampered frames,
                           forged commands, duplicate payloads, persistent
                           freeze) that is absent from the prompt?
  * template rate        - how often is the rationale the fine-tuning template
                           rather than incident-specific text?

No human study is claimed. An expert-rating worksheet is emitted for future
annotation; participants are not fabricated.

Run: python3 -m code.evaluation.final_revision.run_e7_groundedness
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "e7_trustworthiness"
CORPUS_EVAL = Path("code/results/ijcip_revision_r1r2_20260805/lora_eval")
CORPUS = Path("code/finetuning_dataset/revision_eval/eval_corpus.jsonl")

CLASSES = ["none", "fdi", "replay", "command_spoof", "dos", "firmware"]
ACTIONS = ["no_op", "freeze_setpoint", "throttle_ramp",
           "request_ied_revalidation", "isolate_inverter"]

# Evidence claims a rationale can make, and the prompt token that would
# substantiate each. A claim without its marker is unsupported.
EVIDENCE_MARKERS = {
    "tamper": ("tampered", "tamper"),
    "forged command": ("command", "setpoint", "derms"),
    "duplicate/replay": ("replay", "dup"),
    "persistent freeze": ("persistent_freeze", "freeze", "dos_signature"),
    "voltage": ("v_min_pu", "voltage", "pu"),
}
TEMPLATE_RE = re.compile(r"detected \w+ pattern affecting \S+; recommend \w+\.?",
                         re.IGNORECASE)


def analyse(pred_path: Path, prompts: dict[str, str]) -> list[dict]:
    rows = []
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prompt = (prompts.get(r["id"]) or "").lower()
        raw = r.get("raw_output") or ""
        try:
            obj = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception:
            obj = {}
        rationale = str(obj.get("rationale", "")).strip()
        asset = obj.get("affected_asset")
        pred_cls = r.get("pred_class")
        prop = r.get("proposed_action")
        rl = rationale.lower()

        asset_grounded = bool(asset) and str(asset).lower() in prompt
        class_consistent = bool(pred_cls) and pred_cls.replace("_", " ") in \
            rl.replace("_", " ")
        action_consistent = bool(prop) and prop.replace("_", " ") in \
            rl.replace("_", " ")
        unsupported = []
        for claim, markers in EVIDENCE_MARKERS.items():
            claim_made = any(w in rl for w in claim.split("/")[0].split())
            if claim_made and not any(m in prompt for m in markers):
                unsupported.append(claim)
        rows.append({
            "backend": r.get("backend"), "k": r.get("k"), "id": r["id"],
            "family": r.get("family"), "has_rationale": bool(rationale),
            "rationale_chars": len(rationale),
            "asset_grounded": asset_grounded if asset else None,
            "class_consistent": class_consistent,
            "action_consistent": action_consistent,
            "n_unsupported_claims": len(unsupported),
            "unsupported_claims": ";".join(unsupported),
            "is_template_rationale": bool(TEMPLATE_RE.search(rationale)),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompts = {}
    for line in CORPUS.read_text().splitlines():
        rec = json.loads(line)
        prompts[rec["id"]] = next(m["content"] for m in rec["messages"]
                                  if m["role"] == "user")

    rows = []
    for d in sorted(CORPUS_EVAL.iterdir()):
        p = d / "predictions.jsonl"
        if p.exists():
            rows += analyse(p, prompts)
    if not rows:
        print("no predictions found under", CORPUS_EVAL)
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "explanation_groundedness_raw.csv", index=False)

    agg = (df.groupby(["backend", "k"])
             .agg(n=("id", "count"),
                  has_rationale=("has_rationale", "mean"),
                  asset_grounded=("asset_grounded", "mean"),
                  class_consistent=("class_consistent", "mean"),
                  action_consistent=("action_consistent", "mean"),
                  mean_unsupported_claims=("n_unsupported_claims", "mean"),
                  any_unsupported=("n_unsupported_claims",
                                   lambda s: float((s > 0).mean())),
                  template_rationale=("is_template_rationale", "mean"),
                  mean_chars=("rationale_chars", "mean"))
             .reset_index().round(4))
    agg.to_csv(OUT / "explanation_groundedness.csv", index=False)

    fam = (df.groupby("family")
             .agg(n=("id", "count"),
                  asset_grounded=("asset_grounded", "mean"),
                  any_unsupported=("n_unsupported_claims",
                                   lambda s: float((s > 0).mean())),
                  template_rationale=("is_template_rationale", "mean"))
             .reset_index().round(4))
    fam.to_csv(OUT / "explanation_groundedness_by_family.csv", index=False)

    # Expert-rating worksheet: blinded, no participants invented.
    sample = df.sample(min(40, len(df)), random_state=0)[["id", "family"]]
    sample = sample.assign(rating_faithful="", rating_useful="",
                           rater_id="", notes="")
    sample.to_csv(OUT / "expert_rating_worksheet_BLANK.csv", index=False)

    print(agg.to_string(index=False))
    print("\nby family:")
    print(fam.to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
