# Paper ↔ Artifact Map

Mapping from every table and figure in the manuscript ("DER-SafeAgent: A
Runtime-Assurance Architecture for Safe LLM-Assisted Cyber-Physical Incident
Response in Distributed Energy Resources", main text) to its source data,
generation script, output file, and verification method in this repository.

Regeneration entry point (CPU-only):

```bash
make reproduce-paper     # = run_stats_v3 + build_artifacts_v3 + compare_artifacts
```

Outputs land in `artifacts/tables/` and `artifacts/figures/`; the byte-exact
copies of the files included in the submitted manuscript are tracked in
`artifacts/reference/`. "Byte-identical" below means verified equal by
`scripts/compare_artifacts.py` and `tests/test_paper_artifacts.py`.

## Main-text tables

| Paper item | Source data | Generation script | Output file | Verification |
|---|---|---|---|---|
| Table 1 (positioning vs. related systems) | — (qualitative literature table) | — (typeset inline in the manuscript) | — | n/a (no numeric content) |
| Table 2 (evidence provenance / trust boundary) | design of `code/Multi_AI_Agent/telemetry_features.py` | hand-authored descriptive table | `artifacts/reference/tables/table_provenance.tex` | checksum only; content cross-checked by `test_runtime_safe_gate.py` (the provenance claims it makes are enforced as tests) |
| Table 3 (class-aware action policy) | runtime constants in `code/Multi_AI_Agent/safety_projection.py` (single source of truth) + machine-readable export `code/configs/action_policy.yaml` | `scripts/export_action_policy.py` | `artifacts/tables/table_action_policy.tex` | byte-identical to reference; staleness + policy-conformance checks in `tests/test_action_policy.py` |
| Table 4 (70-case adversarial, Qwen) | `code/results/ijcip_final_v3/adversarial/adversarial_summary_qwen.csv` (raw rows: `adversarial_raw_qwen.csv`) | `code/evaluation/final_safeagent_v3/build_artifacts_v3.py` | `artifacts/tables/table_adversarial_v3_qwen.tex` | byte-identical to reference |
| Table 5 (70-case adversarial, Llama) | `…/adversarial_summary_llama.csv` (raw: `adversarial_raw_llama.csv`) | `build_artifacts_v3.py` | `artifacts/tables/table_adversarial_v3_llama.tex` | byte-identical to reference |
| Table 6 (Evidence-Gate coverage, 32-config test) | `code/results/ijcip_final_v3/gate_robustness/gate_v3_summary.csv` | `build_artifacts_v3.py` | `artifacts/tables/table_gate_v3.tex` | byte-identical to reference |
| Table 7 (command-authentication sub-study) | `code/results/ijcip_final/gate_auth/gate_auth_summary.csv` | data: `code/evaluation/final_safeagent_v3/run_gate_auth.py` (`make gate-auth`); table hand-typeset from that CSV | `artifacts/reference/tables/table_gate_auth.tex` | checksum of table + CSV; CSV regenerable on CPU (seeds 2101–2103) |
| Table 8 (49-config end-to-end holdout) | `code/results/ijcip_final_v3/holdout_e2e/holdout_e2e_raw.csv` | `build_artifacts_v3.py` | `artifacts/tables/table_holdout_v3.tex` | byte-identical to reference |
| Table 9 (primary paired statistics, Holm) | `code/results/ijcip_final_v3/statistics/F1_physical.csv` (derived from `holdout_e2e_raw.csv`, RNG seed 20260811) | `code/evaluation/final_safeagent_v3/run_stats_v3.py` → `build_artifacts_v3.py` | `artifacts/tables/table_stats_primary.tex` | statistics CSV regenerates hash-identically; table byte-identical to reference |
| Table 10 (EH estimator robustness, 12-config holdout) | `code/results/ijcip_final_v3/eh_robustness/eh_robustness.csv` | `build_artifacts_v3.py` (data: `run_eh_robustness.py`, CPU) | `artifacts/tables/table_eh_v3.tex` | byte-identical to reference |
| Table 11 (consultation latency) | `code/results/ijcip_final/latency/latency_distributions.csv` (re-derivable value-identically from the four archived `lora_eval/*/predictions.jsonl` via `scripts/derive_latency_v3.py`) | `build_artifacts_v3.py` | `artifacts/tables/table_latency_v3.tex` | byte-identical to reference; derivation verified by `make derive-latency` |

## Main-text figures

| Paper item | Source data | Generation script | Output file | Verification |
|---|---|---|---|---|
| Figure 1 (architecture) | — (TikZ diagram) | editable source `paper/figure/fig_architecture_final.tex` (tracked in this repository; compiled inline by `paper/main.tex`) | not part of the numeric pipeline | n/a (no numeric content) |
| Figure 2 (irrev. proposed vs executed per arm) | `adversarial_summary_qwen.csv` | `build_artifacts_v3.py` | `artifacts/figures/fig_containment_v3.{pdf,png}` | PNG (300 dpi) byte-identical; PDF identical modulo embedded timestamps |
| Figure 3 (gate-open rate by family + benign false-open) | `gate_robustness/gate_v3_summary.csv` | `build_artifacts_v3.py` | `artifacts/figures/fig_gate_v3.{pdf,png}` | PNG byte-identical; PDF identical modulo timestamps |
| Figure 4 (per-config ENS diff vs D0) | `holdout_e2e/holdout_e2e_raw.csv` | `build_artifacts_v3.py` | `artifacts/figures/fig_holdout_ens_diff.{pdf,png}` | PNG byte-identical; PDF identical modulo timestamps |

## Headline in-text numbers (not in a table)

| Claim | Source |
|---|---|
| Structural enumeration: 40,824 states, 0 violations; mutation guards 1008/144/3930 | `code/results/ijcip_final_v3/property_tests/property_test_result.json` (script: `code/evaluation/ijcip_final_v3/property_safety_tests.py`) |
| Pre-action gate invariance: 11,020 ticks, 0 violations | `code/results/ijcip_final_v3/p0_forensics/p0a_result.json` |
| Unshielded Qwen executes isolate on 61/70 | `code/results/ijcip_final_revision/e6_adversarial/e6_raw.csv`, enforced by `test_claim_consistency.py` |
| SLO admission (K=1 from 10 s, K=3 from 30 s) | `code/results/ijcip_final_v3/deadline/slo_admission.csv` (script: `run_deadline_v3.py`) |
| OpenDSS check: 12 configs, 0 non-converged; 17.6 vs 17.7 kWh over the 11 attack configs | `code/results/ijcip_final/opendss_check/opendss_check_{raw,summary}.csv` |
| Groundedness audit: 70–73% templated rationales, 23–30% unsupported evidence (K=1) | `code/results/ijcip_final_revision/e7_trustworthiness/explanation_groundedness.csv` (script: `code/evaluation/final_revision/run_e7_groundedness.py`) |
| Benign-day consultations fall >80% (gated 2 vs ungated 32 calls) | `code/results/ijcip_final_v3/holdout_e2e/holdout_e2e_raw.csv` (`n_llm_calls`, benign rows) |
| Closed-loop coverage definition + per-arm values (Table 8 column) | `code/results/ijcip_final_v3/p0_forensics/p0a_result.json`; definition in `docs/CLAIM_LEDGER.md` |

## Supplementary material

The supplement's tables/figures (scenario matrix, expanded QLoRA evaluation,
K=1 vs K=3, historical det-vs-LLM comparison, 20-config OpenDSS sweep,
configuration-level paired analysis, LoRA confusion/calibration figures,
per-family adversarial breakdown) are produced by the older evaluation
pipelines that also ship in `code/evaluation/` (`build_ijcip_figures.py`,
`build_lora_eval_artifacts.py`, `build_p0_figures.py`,
`build_revision_artifacts.py`). Their canonical inputs under
`code/results/ijcip_revision_r1r2_20260805/lora_eval/` are included and
checksummed; regeneration of the supplementary items was **not**
re-verified for this release (see `REPRODUCIBILITY_REPORT.md`, scope
limitations). The principal results of the paper are the main-text items
above.

## Two hand-authored tables — why

`table_provenance.tex` (Table 2) is a descriptive design table with no
numeric source; `table_gate_auth.tex` (Table 7) was typeset by hand from
`gate_auth_summary.csv` during the final revision. Both ship as
checksummed reference files, and Table 7's underlying CSV regenerates on CPU
with `make gate-auth`. Automating Table 7's TeX emission is a known
improvement item, deliberately not done post hoc for this release to avoid
introducing a generator that never produced the submitted file.
