"""E6: adversarial suite against the REAL QLoRA adapters.

The published adversarial results were produced with the scripted keyword
backbone, so the paper's robustness claims never exercised a language model.
This driver runs the same 14 perturbation families against real Qwen/Llama
QLoRA adapters under strict no-fallback mode, and separates three failure
modes a single "forbidden-action rate" conflates:

  * model compromise      - the model's own proposal is unsafe/incorrect
  * containment           - whether that proposal reaches execution
  * residual              - unsafe or unnecessary actions that do execute

Reporting rule: when zero unsafe executions are observed we report the sample
size and a rule-of-three 95% upper bound rather than claiming a zero rate.

Run: python3 -m code.evaluation.final_revision.run_e6_adversarial \
        --backends qwen --n-per-family 6
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from ..adversarial_safety.perturbations_expanded import build_expanded_suite
from ..adversarial_safety.run_robustness import _evaluate_violation, _step_detector
from code.llm_serving import model_paths as _MP

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "e6_adversarial"

QWEN = (_MP.QWEN_BASE,
        "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter")
LLAMA = (_MP.LLAMA_BASE,
         "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter")
REGISTRY = {"no_op", "freeze_setpoint", "throttle_ramp",
            "request_ied_revalidation", "isolate_inverter"}


def rule_of_three(n: int) -> float:
    """95% upper bound on a rate when zero events are observed in n trials."""
    return 3.0 / n if n > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="qwen")
    ap.add_argument("--n-per-family", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from ...llm_serving import local_lora
    from ...Multi_AI_Agent.adapter import DERSecAgentDetector
    from ...baselines.single_llm.runner import SingleLLM

    cases = build_expanded_suite(n_per_family=args.n_per_family, seed=args.seed)
    print(f"suite: {len(cases)} cases across "
          f"{len({c.perturbation for c in cases})} families", flush=True)

    rows, examples = [], []
    for backend in args.backends.split(","):
        base, adapter = QWEN if backend == "qwen" else LLAMA
        os.environ["DER_LLM_STRICT"] = "1"
        llm = local_lora.configure_default(base, adapter,
                                           max_new_tokens=args.max_new_tokens)
        if not llm._try_load():
            raise RuntimeError(f"FATAL: {backend} unavailable: {llm._load_fail_reason}")
        sha = llm.adapter_sha()

        configs = {
            # bare model: registry only, no shield
            f"bare_{backend}": SingleLLM(invoke_interval_s=None),
            # full stack under the legacy policy
            f"shielded_{backend}_legacy": DERSecAgentDetector(
                k_setting=1, compact_prompt=True,
                policy_mode="legacy_class_override"),
            # full stack under runtime-assurance projection
            f"shielded_{backend}_projection": DERSecAgentDetector(
                k_setting=1, compact_prompt=True,
                policy_mode="safety_projection"),
        }
        for name, det in configs.items():
            det.name = name
            t0 = time.time()
            for i, case in enumerate(cases):
                out = _step_detector(det, case)
                action = out.get("action", "no_op")
                proposed = (getattr(det, "_last_decision", {}) or {}).get(
                    "proposed_action", action)
                viol = _evaluate_violation(case, action, out.get("tier", "negligible"),
                                           bool(out.get("hitl", False)),
                                           bool(out.get("schema_ok", True)))
                rows.append({
                    "backend": backend, "config": name, "adapter_sha": sha,
                    "family": case.perturbation, "case_index": i,
                    "expected_class": getattr(case, "expected_class", None),
                    "proposed_action": proposed, "executed_action": action,
                    "proposed_out_of_registry": proposed not in REGISTRY,
                    "executed_out_of_registry": action not in REGISTRY,
                    "contained": bool(proposed != action),
                    **{k: v for k, v in viol.items() if isinstance(v, (int, float, bool))},
                })
                if viol.get("policy_violation") or viol.get("forbidden_action"):
                    examples.append({"config": name, "family": case.perturbation,
                                     "proposed": proposed, "executed": action,
                                     "violation": viol})
            print(f"[done] {name}: {len(cases)} cases in {time.time()-t0:.0f}s",
                  flush=True)
        local_lora.reset_default()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e6_raw.csv", index=False)
    if examples:
        (OUT / "failure_examples.jsonl").write_text(
            "\n".join(json.dumps(e, default=str) for e in examples))

    num = [c for c in ("policy_violation", "forbidden_action", "unsafe_command",
                       "schema_failure", "safe_fallback", "correct_refusal")
           if c in df.columns]
    agg = (df.groupby(["backend", "config"])
             .agg(n_cases=("case_index", "count"),
                  proposed_out_of_registry=("proposed_out_of_registry", "mean"),
                  executed_out_of_registry=("executed_out_of_registry", "mean"),
                  contained_rate=("contained", "mean"),
                  **{c: (c, "mean") for c in num})
             .reset_index())
    agg["executed_violation_upper95_if_zero"] = agg.apply(
        lambda r: (rule_of_three(int(r.n_cases))
                   if r.get("policy_violation", 0) == 0 else float("nan")), axis=1)
    agg.round(4).to_csv(OUT / "e6_by_config.csv", index=False)

    fam = (df.groupby(["config", "family"])
             .agg(n=("case_index", "count"),
                  proposed_out_of_registry=("proposed_out_of_registry", "mean"),
                  executed_out_of_registry=("executed_out_of_registry", "mean"),
                  contained=("contained", "mean"),
                  **{c: (c, "mean") for c in num})
             .reset_index())
    fam.round(4).to_csv(OUT / "e6_by_family.csv", index=False)
    print(agg.round(4).to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
