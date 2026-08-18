#!/usr/bin/env bash
# Fast CPU-only smoke test (~1 minute, no GPU, no LLM, no OpenDSS).
# Checks: imports, configuration loading, schema validation, the runtime-safe
# FeatureView, the Evidence Gate, the safety-projection shield, the action
# registry, audit records, and a miniature end-to-end episode.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

echo ">> [1/4] package imports and configuration loading"
"$PY" -m pytest tests/test_repo_smoke.py -q

echo ">> [2/4] runtime-safety invariants (no ground-truth leakage, shield)"
"$PY" -m pytest \
    code/Multi_AI_Agent/test_runtime_safe_gate.py \
    code/Multi_AI_Agent/test_safety_projection.py \
    code/llm_serving/test_strict_mode.py -q

echo ">> [3/4] canonical-result metric invariants"
"$PY" -m pytest \
    code/evaluation/final_safeagent_v3/test_metric_invariants.py \
    code/evaluation/final_safeagent_v3/test_claim_consistency.py -q

echo ">> [4/4] miniature stochastic episode on the StubFeeder harness"
"$PY" -m pytest code/simulation/test_harness_smoke.py \
    code/simulation/test_stochastic_episodes.py -q

echo "SMOKE TEST PASSED"
