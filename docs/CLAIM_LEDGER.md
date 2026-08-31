# Claim Ledger — DER-SafeAgent manuscript vs. released artifacts

Audit date: 2026-08-29. Every principal reported number is traced to its
canonical raw source in this repository and re-verified from that source
(commands below were executed; see `RELEASE_AUDIT.md` for the environment).
Statuses: **VERIFIED** (recomputed from released raw data and equal to the
manuscript), **VERIFIED_WITH_LIMITATION** (recomputed and equal, with a
documented caveat), **NOT_REPRODUCIBLE_WITHOUT_GPU** (the producing
computation needs GPU + base weights; the shipped raw data is checksummed
and all derived numbers recompute from it). No claim in the manuscript is
currently UNVERIFIED_MISSING_SOURCE or UNVERIFIED_MISMATCH.

Abbreviations: `R = code/results/ijcip_final_v3`,
`RL = code/results/ijcip_final` (legacy tag, gate-auth/latency/OpenDSS),
`RS = code/results/ijcip_final_safeagent_20260810` (estimator),
`RR = code/results/ijcip_final_revision` (groundedness). Seeds: adversarial
suite seed 0; statistics RNG 20260811; fine-tuning seed 0. Verification of
generated tables/figures = `make reproduce-paper` (byte-identical against
`artifacts/reference/`); the whole ledger is re-checkable with
`make test && make reproduce-paper`.

| # | Claim (paper location) | Reported value | Raw source | Generation script | Verification | Status |
|---|---|---|---|---|---|---|
| 1 | 70-case adversarial suite (§5.1, §5.2) | 14 families × 5 cases, per backbone | `R/adversarial/adversarial_raw_{qwen,llama}.csv` (70 rows/arm) | `run_adversarial_v3.py` (suite: `perturbations_expanded.build_expanded_suite(n_per_family=5, seed=0)`) | `tests/test_action_policy.py` re-builds the suite (70 cases); raw rows counted | VERIFIED |
| 2 | Qwen full arch. irreversible execution 0/70 (Tab. 4) | 0/70, rate 0.000 | `R/adversarial/adversarial_summary_qwen.csv` (`final`) | `run_adversarial_v3.py` | table byte-identical; `test_metric_invariants.py` | VERIFIED (execution itself: NOT_REPRODUCIBLE_WITHOUT_GPU) |
| 3 | Llama full arch. irreversible execution 0/70 (Tab. 5) | 0/70 | `R/adversarial/adversarial_summary_llama.csv` (`final`) | same | same | VERIFIED (same caveat) |
| 4 | Qwen unshielded irreversible execution 61/70 (abstract, §5.2) | 61/70 = 0.871 | `R/adversarial/adversarial_summary_qwen.csv` (`bare_llm`) | same | recomputed 61; `test_e6_61_of_70_matches_canonical` | VERIFIED |
| 5 | Llama unshielded 18/70 (abstract, §5.2) | 18/70 = 0.257 | `R/adversarial/adversarial_summary_llama.csv` (`bare_llm`) | same | recomputed 18 | VERIFIED |
| 6 | Per-backbone Clopper–Pearson upper bound (abstract, §5.2) | 0.051 | `R/statistics/F2_adversarial.csv` (`irrev_exec_upper95` = 0.0513) | `run_stats_v3.py` | `make reproduce-stats` hash-identical | VERIFIED |
| 7 | Deterministic enumeration 40,824 states (§5.2) | 40,824 combinations | `R/property_tests/property_test_result.json` (`grid_size`) | `code/evaluation/property_tests/` (`make property-tests`) | re-run in E2 content-identical (see REPRODUCIBILITY_REPORT) | VERIFIED |
| 8 | Zero invariant violations (§5.2) | 0 violations, all properties hold | same (`property_violations: 0`) | same | re-run content-identical | VERIFIED |
| 9 | Guard-removal mutations 1008 / 144 / 3930 (§5.2) | irreversible-escalation 1008; class-avoidance 144; inaction-veto 3930 | same (`mutation_tests`) | same | re-run content-identical | VERIFIED |
| 10 | Evidence-Gate family counts (Tab. 6) | spoof 4/4, replay 4/4, DoS 4/4, FDI-high 1/1, FDI-med 0/1, FDI-low 0/1 | `R/gate_robustness/gate_v3_summary.csv` | `run_gate_robustness.py` (`make gate-evaluate`) | re-run byte-identical (E2) | VERIFIED |
| 11 | Benign false-open 4/16 = 0.25 [0.07, 0.52] (§5.3, Tab. 6) | 4/16, CI [0.073, 0.524] | same (`benign_false_open`) | same | re-run byte-identical | VERIFIED |
| 12 | Command-auth sub-study counts (Tab. 7) | naive 3/3 & 2/2; auth-required 3/3 & 0/2; no-auth 3/3 & 2/2 | `RL/gate_auth/gate_auth_summary.csv` | `run_gate_auth.py` (`make gate-auth`) | re-run byte-identical (E2); Table 7 hand-typeset from this CSV | VERIFIED_WITH_LIMITATION (tex hand-typeset, checksummed) |
| 13 | 49-configuration holdout (§5.1, §5.4) | 49 configs, hash-frozen | `code/configs/ijcip_final_v3/holdout_v3_manifest.csv` + `holdout_v3_freeze.json` | `gen_holdout_v3.py` | `test_holdout_config_count_matches_manifest`; freeze digest pinned | VERIFIED |
| 14 | 33 attack + 16 benign (Tab. 8) | 33/16 | same manifest | same | `test_repo_smoke.py::test_holdout_manifest_composition` | VERIFIED |
| 15 | 3 stochastic episodes per configuration (Tab. 8) | 3 | `R/holdout_e2e/holdout_e2e_raw.csv` (episode ∈ {0,1,2}) | `run_holdout_e2e.py` | recounted from raw (1,176 rows = 49×3×8) | VERIFIED |
| 16 | D0 attack ENS 0.352 (Tab. 8) | 0.352 kWh | `R/holdout_e2e/holdout_e2e_raw.csv` | `build_artifacts_v3.py` | table byte-identical; D0/OPROJ rows re-simulated value-identically (E2, `make holdout-cpu`) | VERIFIED |
| 17 | SafeProj-Q ENS 0.537 (Tab. 8) | 0.537 | same (arm `QPROJ`) | same | table byte-identical | VERIFIED (LLM arm rows: NOT_REPRODUCIBLE_WITHOUT_GPU) |
| 18 | SafeProj-L ENS 0.750 (Tab. 8) | 0.750 | same (`LPROJ`) | same | same | VERIFIED (same caveat) |
| 19 | ModelOnly-Q/L ENS 8.484 (Tab. 8) | 8.484 both; physical trajectories byte-identical | same (`bareQ`,`bareL`) | same | paired vectors recomputed element-identical (basis of the Table 9 CI fix) | VERIFIED |
| 20 | OracleClass-Proj ENS 0.067 (Tab. 8) | 0.067 | same (`OPROJ`) | same | table byte-identical; re-simulated value-identically | VERIFIED |
| 21 | Table 9 paired stats (diffs, CIs, p, Holm) | e.g. OracleClass −0.285 [−0.69, −0.05], raw p 0.015, Holm 0.107; ModelOnly +8.133 [+4.25, +12.46] | `R/statistics/F1_physical.csv` | `run_stats_v3.py` (seed 20260811; **shared resampling draws across contrasts — protocol amended 2026-08-29** so the identical ModelOnly-Q/L vectors get identical CIs) | `make reproduce-stats` hash-identical; table byte-identical | VERIFIED_WITH_LIMITATION (amended CRN protocol; original 2026-08-11 CIs differed in the last digits between the identical ModelOnly arms) |
| 22 | OpenDSS 12-configuration check (§5.4) | 12 configs re-run, D0 & OracleClass-Proj | `RL/opendss_check/opendss_check_raw.csv` (12 configs × 2 arms) | `run_opendss_check.py` (`make opendss-check`) | recounted from raw | VERIFIED (re-execution needs `opendssdirect`) |
| 23 | Zero non-converged solves (§5.4) | 0 | same (`n_nonconverged` all 0) | same | recomputed sum = 0 | VERIFIED |
| 24 | OpenDSS curtailment 17.6 vs 17.7 kWh, vmin 0.975 pu (§5.4) | 17.577 / 17.671 over the 11 attack configs (12th = benign control; stated in the manuscript), min vmin 0.9753 | `RL/opendss_check/opendss_check_summary.csv` + raw | same | recomputed from raw | VERIFIED |
| 25 | Estimator 12-configuration holdout (Tab. 10) | 12 configs, branched counterfactual rollouts | `RS/p0a_estimator/{rollouts,selections,metrics}_holdout.*` | `code/evaluation/final_safeagent/p0a_*` (frozen split `impact_estimator_holdout.csv`) | metrics recomputed from shipped JSON/CSV | VERIFIED (rollout generation NOT RE-RUN; shipped + checksummed) |
| 26 | E60 top-1 0.33 (0.40 over 10 conditions), mean regret 76.3; EH 12/12 zero-regret, retained under perturbations; MAE ≈26 (§5.4, Tab. 10) | E60 top1 0.333 nominal / 0.40 mean; regret 76.34; EH top1 = 1.0 in all 10 conditions; EH MAE(curt) 26.3 | `RS/p0a_estimator/metrics_holdout.json`; `R/eh_robustness/eh_robustness.csv` | `run_eh_robustness.py` (`make eh-robustness`) | recomputed; EH-robustness re-run byte-identical (E2) | VERIFIED |
| 27 | Latency K=1 5.45 s (Qwen) / 5.62 s (Llama); K=3 16.8–17.4 s (§5.5, Tab. 11) | 5.446 / 5.618; 16.825 / 17.448 | `R/deadline/latency_distributions.csv`, re-derived from 4 archived `lora_eval/*/predictions.jsonl` | `derive_latency_v3.py`; `run_deadline_v3.py` | `make derive-latency` value-identical; table byte-identical | VERIFIED_WITH_LIMITATION (E1 hardware-specific) |
| 28 | Strict serving, zero fallbacks (§5.4, Tab. 8) | 0 fallback rows in every LLM result | `n_fallbacks`/`result_source` columns in all raw CSVs | serving assertions in `code/llm_serving/local_lora.py` | `test_no_serving_fallback_in_any_llm_result`; `test_strict_mode.py` | VERIFIED |
| 29 | Rationale templating 70–73% (§6.2) | Llama 69.9% → Qwen 72.6% (K=1) | `RR/e7_trustworthiness/explanation_groundedness.csv` (**added to the release 2026-08-29**) | `code/evaluation/final_revision/run_e7_groundedness.py` | recomputed from shipped CSV | VERIFIED |
| 30 | Unsupported-evidence rationales 23–30% (§6.2) | Llama 22.6% → Qwen 30.1% (K=1, `any_unsupported`) | same | same | recomputed | VERIFIED |
| 31 | Benign-day consultations fall >80% (§6.5) | gated arms 2 vs ungated 32 calls on the 16 benign configs (−93.75%) | `R/holdout_e2e/holdout_e2e_raw.csv` (`n_llm_calls`, benign rows) | `run_holdout_e2e.py` | recomputed from raw | VERIFIED |
| 32 | QLoRA adapter SHA-256 (§5.1, availability) | Qwen `0c9dec242e2f…`, Llama `9a2b8436ce9c…` | `code/finetuning/results/2026*/adapter/adapter_model.safetensors` (Git LFS) | training: `code/finetuning/train.py` (seed 0) | `make verify` digests; run manifests record the same `adapter_sha` per call | VERIFIED (adapter file hash; re-training not expected to be bit-identical) |
| 33 | Prompt/configuration hashes recorded per call (§4.4) | prompt SHA + config hash in every serving record | `predictions.jsonl` runs; freeze JSONs in `code/configs/` | serving layer + freeze scripts | `make verify` (freeze-JSON self-hash pins) | VERIFIED |
| 34 | Fine-tuning corpus 200 examples, 161/21/18 split (§5.1, availability) | 161 + 21 + 18 = 200 | `code/finetuning_dataset/processed/{train,val,test}.jsonl` | corpus builders in `code/finetuning_dataset/` | line counts recomputed; `test_finetuning_corpus_schema_and_split` | VERIFIED |
| 35 | Zero exact/normalized duplicates, eval vs. training (§5.1) | exact 0, normalized 0, max 5-gram overlap 0.0 | `R/adversarial/leakage_summary.json`, `leakage_report.csv` | leakage check inside `run_adversarial_v3.py` | recomputed from shipped JSON | VERIFIED |

## Table 8 closed-loop coverage — metric definition (audit item)

`Closed-loop coverage` (Table 8) is **not** the detector-side gate recall of
Table 6. Definition (from `run_holdout_e2e.py` + `build_artifacts_v3.py`):
per episode, the boolean `gate_open_covered` = "the Evidence Gate opened at
≥1 tick inside the attack window *of that arm's own closed-loop
trajectory*"; episodes are averaged within a configuration, then over the
33 attack configurations (denominator 33; the 16 benign configurations are
excluded; N/A for the ungated bare arms). It is arm-dependent because each
arm's executed actions feed back into telemetry and the sustained-evidence
window (D0 first gate-open ≈20 s vs. L1 ≈111 s). The *pre-action* gate
decision is model-independent: `R/p0_forensics/p0a_result.json` records the
invariance test — across gated arms, gate-open at every attack-window tick
strictly before the earliest first-action tick is identical: **11,020 ticks
compared, 0 violations** — and the per-arm coverage values in that file
(D0 0.949, Q1/QPROJ 0.657, L1 0.657, LPROJ 0.495, OPROJ 0.960) are exactly
the Table 8 column. Status: **VERIFIED** (metric is well-defined, correctly
labelled in the caption, and reproduced from raw).

## Table 9 ModelOnly CI discrepancy — resolution (audit item)

The 2026-08-11 statistics run consumed one sequential RNG stream across
contrasts, so the element-identical ModelOnly-Q and ModelOnly-L difference
vectors received different bootstrap draws ([+4.25, +12.50] vs
[+4.28, +12.53]). Re-analysis (no new experiment): resampling draws are now
a function of the family seed (20260811) and the sample size only, shared
across all contrasts (common random numbers). Identical vectors now yield
identical CIs ([+4.25, +12.46], both arms). All qualitative conclusions are
unchanged (same Holm significance pattern); the manuscript's Table 9 and
the three body-text p-values were updated to the amended output.
`ModelOnly` remains labelled diagnostic, not a matched shield ablation, in
the caption, §5.4, and Fig. 3's caption.
