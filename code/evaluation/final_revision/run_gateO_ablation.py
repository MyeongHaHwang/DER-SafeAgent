"""Gate O: ablation that identifies actual marginal contributions.

Why the published ablation failed the gate
------------------------------------------
The main-scenario ablation produced eight byte-identical rows. That is not a
finding about the architecture; it is a property of the evaluation. Under the
scripted backbone with the class-override policy, the executed action is fully
determined by the predicted class, so removing the Caution Agent, the Energy
Impact estimator, incident memory or self-consistency cannot change anything.
An ablation can only measure a component's contribution on inputs where that
component is able to fire.

What this driver does instead
-----------------------------
Ablations are evaluated on the **adversarial suite with a real QLoRA adapter**,
which is exactly the regime where the safety layers are load-bearing (E6 showed
the bare model reaches a 0.657 policy-violation rate there). For every ablation
we record the model's own proposal *and* the executed action, so the table
reports **which layer prevented the unsafe execution** rather than an aggregate
zero.

Marginal contribution is reported as the delta against FULL: a component whose
removal changes nothing on inputs that can activate it genuinely contributes
nothing, and we say so.

Run: python3 -m code.evaluation.final_revision.run_gateO_ablation \
        --n-per-family 3 --backend qwen
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from ..adversarial_safety.perturbations_expanded import build_expanded_suite
from ..adversarial_safety.run_robustness import _evaluate_violation, _step_detector
from code.llm_serving import model_paths as _MP

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "gateO_ablation"
QWEN = (_MP.QWEN_BASE,
        "code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter")
LLAMA = (_MP.LLAMA_BASE,
         "code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter")
REGISTRY = {"no_op", "freeze_setpoint", "throttle_ramp",
            "request_ied_revalidation", "isolate_inverter"}
IRREVERSIBLE = "isolate_inverter"

# Ablation flags supported by DERSecAgentDetector, plus the policy axis.
ABLATIONS = [
    ("full", "full", "legacy_class_override"),
    ("no_caution_agent", "no_caution_agent", "legacy_class_override"),
    ("no_energy_impact_agent", "no_energy_impact_agent", "legacy_class_override"),
    ("no_class_aware_avoidance", "no_class_aware_avoidance", "legacy_class_override"),
    ("no_self_consistency", "no_self_consistency", "legacy_class_override"),
    ("no_incident_memory", "no_incident_memory", "legacy_class_override"),
    ("no_hitl_fallback", "no_hitl_fallback", "legacy_class_override"),
    ("deterministic_only", "deterministic_only", "legacy_class_override"),
    # policy axis: the shield construction itself
    ("full_safety_projection", "full", "safety_projection"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--n-per-family", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from ...llm_serving import local_lora
    from ...Multi_AI_Agent.adapter import DERSecAgentDetector

    base, adapter = QWEN if args.backend == "qwen" else LLAMA
    os.environ["DER_LLM_STRICT"] = "1"
    llm = local_lora.configure_default(base, adapter, max_new_tokens=args.max_new_tokens)
    if not llm._try_load():
        raise RuntimeError(f"FATAL: {args.backend} unavailable: {llm._load_fail_reason}")
    sha = llm.adapter_sha()

    cases = build_expanded_suite(n_per_family=args.n_per_family, seed=args.seed)
    print(f"suite: {len(cases)} cases, {len({c.perturbation for c in cases})} families; "
          f"{len(ABLATIONS)} ablations -> {len(cases)*len(ABLATIONS)} model-consulted "
          f"decisions", flush=True)

    rows = []
    for label, flag, mode in ABLATIONS:
        det = DERSecAgentDetector(ablation=flag, k_setting=1, compact_prompt=True,
                                  policy_mode=mode)
        det.name = f"abl_{label}"
        t0 = time.time()
        for i, case in enumerate(cases):
            out = _step_detector(det, case)
            action = out.get("action", "no_op")
            last = getattr(det, "_last_decision", {}) or {}
            proposed = last.get("proposed_action", action)
            viol = _evaluate_violation(case, action, out.get("tier", "negligible"),
                                       bool(out.get("hitl", False)),
                                       bool(out.get("schema_ok", True)))
            rows.append({
                "ablation": label, "flag": flag, "policy_mode": mode,
                "backend": args.backend, "adapter_sha": sha,
                "family": case.perturbation, "case_index": i,
                "proposed_action": proposed, "executed_action": action,
                "proposed_irreversible": proposed == IRREVERSIBLE,
                "executed_irreversible": action == IRREVERSIBLE,
                "executed_out_of_registry": action not in REGISTRY,
                "layer_intervened": bool(proposed != action),
                "hitl": bool(out.get("hitl", False)),
                **{k: v for k, v in viol.items() if isinstance(v, (int, float, bool))},
            })
        print(f"[done] {label}: {len(cases)} cases in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gateO_raw.csv", index=False)

    num = [c for c in ("policy_violation", "forbidden_action", "unsafe_command",
                       "schema_failure", "safe_fallback", "correct_refusal")
           if c in df.columns]
    agg = (df.groupby("ablation")
             .agg(n=("case_index", "count"),
                  proposed_irreversible=("proposed_irreversible", "sum"),
                  executed_irreversible=("executed_irreversible", "sum"),
                  layer_intervened=("layer_intervened", "mean"),
                  hitl_rate=("hitl", "mean"),
                  executed_out_of_registry=("executed_out_of_registry", "mean"),
                  **{c: (c, "mean") for c in num})
             .reset_index())

    # Marginal contribution = delta against FULL on inputs that can activate it.
    base_row = agg[agg.ablation == "full"].iloc[0]
    for c in ["policy_violation", "executed_irreversible", "layer_intervened"]:
        if c in agg.columns:
            agg[f"delta_{c}_vs_full"] = agg[c] - base_row[c]
    agg["contributes"] = agg.apply(
        lambda r: "no measurable contribution" if (
            r.ablation != "full"
            and abs(r.get("delta_policy_violation_vs_full", 0)) < 1e-9
            and abs(r.get("delta_executed_irreversible_vs_full", 0)) < 1e-9)
        else ("reference" if r.ablation == "full" else "changes outcome"), axis=1)
    agg.round(4).to_csv(OUT / "gateO_by_ablation.csv", index=False)

    fam = (df.groupby(["ablation", "family"])
             .agg(n=("case_index", "count"),
                  proposed_irreversible=("proposed_irreversible", "sum"),
                  executed_irreversible=("executed_irreversible", "sum"),
                  layer_intervened=("layer_intervened", "mean"),
                  **{c: (c, "mean") for c in num})
             .reset_index())
    fam.round(4).to_csv(OUT / "gateO_by_family.csv", index=False)

    cols = ["ablation", "n", "proposed_irreversible", "executed_irreversible",
            "layer_intervened", "hitl_rate"] + num + ["contributes"]
    print(agg[[c for c in cols if c in agg.columns]].round(3).to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
