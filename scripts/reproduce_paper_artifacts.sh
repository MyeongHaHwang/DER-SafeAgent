#!/usr/bin/env bash
# Regenerate the manuscript's numeric tables (Tables 4-6, 8-11) and figures
# (Figures 2-4) from the canonical raw result files, then compare against the
# reference copies that match the submitted manuscript byte-for-byte.
#
# CPU-only; requires numpy/pandas/scipy/matplotlib. No GPU, no LLM.
# SOURCE_DATE_EPOCH pins the PDF timestamp metadata for determinism.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

echo ">> [1/3] regenerate paired statistics (fixed RNG seed 20260811)"
"$PY" -m code.evaluation.final_safeagent_v3.run_stats_v3

echo ">> [2/4] regenerate tables and figures -> artifacts/"
SOURCE_DATE_EPOCH=0 "$PY" -m code.evaluation.final_safeagent_v3.build_artifacts_v3

echo ">> [3/4] regenerate the action-policy export and manuscript table"
"$PY" scripts/export_action_policy.py

echo ">> [4/4] compare regenerated artifacts against the manuscript reference"
"$PY" scripts/compare_artifacts.py

echo "PAPER-ARTIFACT REPRODUCTION PASSED"
