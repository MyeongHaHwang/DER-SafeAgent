#!/usr/bin/env bash
# Full reproduction ladder. Stages are ordered cheapest-first; the GPU stages
# at the end re-run LLM inference and are OPTIONAL for verifying the
# manuscript artifacts (Tables/Figures regenerate from archived results).
#
#   CPU stages: tests, smoke, artifact verification, paper-artifact
#               regeneration, gate evaluation, EH robustness, deadline
#               re-analysis, latency derivation, deterministic holdout arms.
#   GPU stages: pass --gpu to include the six LLM holdout arms and the
#               70-case adversarial suite (Qwen2.5-7B + Llama-3.1-8B QLoRA;
#               ~10 GB VRAM; many hours — see docs/EXPERIMENTS.md).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_GPU=0
[ "${1:-}" = "--gpu" ] && RUN_GPU=1

make test
make smoke
make verify
make reproduce-paper
make gate-evaluate
make gate-auth
make eh-robustness
make deadline
make derive-latency
make holdout-cpu

if [ "$RUN_GPU" = "1" ]; then
    echo ">> GPU stages: strict LLM serving (DER_LLM_STRICT=1, no fallback output)"
    make holdout-llm
    make adversarial
    make reproduce-stats
    "${PY:-python3}" -m pytest code/evaluation/final_safeagent_v3/test_metric_invariants.py -q
else
    echo ">> GPU stages skipped (pass --gpu to include; see docs/EXPERIMENTS.md)"
fi

echo "REPRODUCTION LADDER COMPLETE"
