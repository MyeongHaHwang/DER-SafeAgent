# DER-SafeAgent reproducibility package — entry points.
# All targets run from the repository root (module paths are `code.…`).
# See README.md and docs/REPRODUCIBILITY.md for details.

PY ?= python3
export PY

.PHONY: help setup setup-llm setup-dss test smoke verify reproduce-paper \
        reproduce-stats derive-latency environment policy-export hygiene \
        latex release-check \
        gate-calibrate gate-evaluate gate-auth eh-robustness deadline \
        holdout-freeze holdout-cpu holdout-llm adversarial opendss-check \
        full-experiment clean-artifacts

help:
	@echo "Verification (CPU-only):"
	@echo "  setup            pip install core dependencies (no GPU stack)"
	@echo "  test             run the full unit/safety test suite"
	@echo "  smoke            fast smoke test (imports, schemas, gate, shield)"
	@echo "  verify           SHA-256 verification of all released artifacts"
	@echo "  reproduce-paper  regenerate manuscript tables/figures from the"
	@echo "                   canonical raw results and compare to reference"
	@echo ""
	@echo "Individual experiment stages (CPU):"
	@echo "  gate-calibrate gate-evaluate gate-auth eh-robustness deadline"
	@echo "  derive-latency holdout-cpu"
	@echo ""
	@echo "Full model-in-the-loop re-execution (GPU; see docs/EXPERIMENTS.md):"
	@echo "  holdout-llm adversarial full-experiment"
	@echo ""
	@echo "  opendss-check    static AC power-flow check (needs opendssdirect)"
	@echo "  environment      record the local execution environment"

# ---- setup ----------------------------------------------------------------

setup:
	$(PY) -m pip install -r requirements.txt

setup-llm:  ## GPU / LLM stack for full re-execution
	$(PY) -m pip install torch>=2.4 "transformers>=4.46,<5" "peft>=0.13,<0.15" \
	    "accelerate>=1.0,<2" bitsandbytes>=0.43

setup-dss:  ## OpenDSS backend for the power-flow check
	$(PY) -m pip install "opendssdirect.py>=0.8"

# ---- verification (CPU) ---------------------------------------------------

test:
	$(PY) -m pytest code/ tests/ -q

smoke:
	bash scripts/smoke_test.sh

verify:
	$(PY) scripts/verify_artifacts.py

reproduce-paper:
	bash scripts/reproduce_paper_artifacts.sh

reproduce-stats:
	$(PY) -m code.evaluation.final_safeagent_v3.run_stats_v3

derive-latency:
	$(PY) scripts/derive_latency_v3.py

environment:
	$(PY) scripts/collect_environment.py

policy-export:  ## regenerate action_policy.yaml + the manuscript policy table
	$(PY) scripts/export_action_policy.py

hygiene:  ## secret / stale-wording / large-file / absolute-path scan
	$(PY) scripts/repo_hygiene.py

latex:  ## build the manuscript from the tracked source (needs a TeX engine)
	cd paper && tectonic main.tex

release-check:  ## full CPU release gate: everything must pass (non-zero on failure)
	$(PY) -c "import numpy, pandas, scipy, matplotlib, yaml; print('deps ok')"
	$(PY) -m compileall -q code tests scripts
	$(PY) -m pytest code/ tests/ -q
	bash scripts/smoke_test.sh
	$(PY) scripts/verify_artifacts.py
	bash scripts/reproduce_paper_artifacts.sh
	$(PY) scripts/derive_latency_v3.py
	$(PY) scripts/repo_hygiene.py
	@if command -v tectonic >/dev/null 2>&1; then \
	    (cd paper && tectonic main.tex >/dev/null) && echo "latex: PASS"; \
	else echo "latex: SKIPPED (no tectonic on PATH — install or run 'make latex' elsewhere)"; fi
	@echo "RELEASE CHECK PASSED"

# ---- CPU experiment stages ------------------------------------------------

property-tests:  ## structural enumeration (40,824 states) + guard-removal mutations
	$(PY) -m code.evaluation.ijcip_final_v3.property_safety_tests

gate-calibrate:  ## re-derive the frozen Evidence-Gate operating point (dev data)
	$(PY) -m code.evaluation.final_safeagent_v3.run_gate_robustness --calibrate

gate-evaluate:  ## one-shot evaluation on the frozen 32-config gate test set
	$(PY) -m code.evaluation.final_safeagent_v3.run_gate_robustness --evaluate

gate-auth:
	$(PY) -m code.evaluation.final_safeagent_v3.run_gate_auth

eh-robustness:
	$(PY) -m code.evaluation.final_safeagent_v3.run_eh_robustness

deadline:
	$(PY) -m code.evaluation.final_safeagent_v3.run_deadline_v3

holdout-freeze:  ## regenerate the 49-config holdout scenario set + manifest
	$(PY) -m code.evaluation.final_safeagent_v3.gen_holdout_v3

holdout-cpu:  ## deterministic + oracle arms of the end-to-end holdout
	$(PY) -m code.evaluation.final_safeagent_v3.run_holdout_e2e --arms D0,OPROJ

# ---- GPU experiment stages (strict serving; no fallback output) -----------

holdout-llm:
	DER_LLM_STRICT=1 $(PY) -m code.evaluation.final_safeagent_v3.run_holdout_e2e --arms Q1,QPROJ
	DER_LLM_STRICT=1 $(PY) -m code.evaluation.final_safeagent_v3.run_holdout_e2e --arms L1,LPROJ
	DER_LLM_STRICT=1 $(PY) -m code.evaluation.final_safeagent_v3.run_holdout_e2e --arms bareQ,bareL

adversarial:
	DER_LLM_STRICT=1 $(PY) -m code.evaluation.final_safeagent_v3.run_adversarial_v3 --backend qwen
	DER_LLM_STRICT=1 $(PY) -m code.evaluation.final_safeagent_v3.run_adversarial_v3 --backend llama

opendss-check:
	$(PY) -m code.evaluation.final_safeagent_v3.run_opendss_check

full-experiment: test holdout-cpu holdout-llm adversarial eh-robustness \
                 gate-evaluate gate-auth deadline
	$(PY) -m code.evaluation.final_safeagent_v3.run_stats_v3
	$(PY) -m code.evaluation.final_safeagent_v3.build_artifacts_v3
	$(PY) -m pytest code/evaluation/final_safeagent_v3/test_metric_invariants.py -q

clean-artifacts:
	rm -rf artifacts/tables artifacts/figures artifacts/processed
