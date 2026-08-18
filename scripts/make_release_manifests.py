#!/usr/bin/env python3
"""(Maintainer tool) Regenerate the release manifests.

Produces:
  * ``data/manifests/sha256_all.json``    — SHA-256 of every canonical data
    file shipped with the release (configs, raw results, corpus, adapters,
    reference artifacts).
  * ``data/manifests/release_manifest.json`` — the curated per-artifact
    manifest with provenance metadata.

Run only when the canonical artifact set itself legitimately changes (e.g. a
new experiment run replaces the archived results). NEVER run this to make a
failed verification pass: a checksum mismatch means the data changed and must
be investigated first (see docs/REPRODUCIBILITY.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/manifests"

HASHED_TREES = [
    "code/configs",
    "code/results",
    "code/finetuning/results",
    "code/finetuning_dataset/processed",
    "code/finetuning_dataset/raw",
    "code/datasets/unsw_mg24/raw",
    "artifacts/reference",
]

MIT = "MIT (repository license)"
CC0 = "CC0-1.0 (synthetic)"

# (glob, description, source, generation_script, configuration, split, seed,
#  paper_reference, license, redistribution_status)
CURATED = [
    # ---- frozen protocols / configurations --------------------------------
    ("code/configs/ijcip_final_v3/evidence_gate_v3_frozen.json",
     "Frozen Evidence-Gate operating point (p_hard=2, soft channel disabled, z_soft=3.0), self-hashing",
     "calibrated on the 25-config development set, frozen before the one-shot test",
     "code/evaluation/final_safeagent_v3/run_gate_robustness.py --calibrate",
     "dev benign + dev attack manifests", "dev", "0",
     "Sec. 5 (RQ2), Table 5", MIT, "released"),
    ("code/configs/ijcip_final_v3/holdout_v3_manifest.csv",
     "Hash-frozen 49-configuration end-to-end holdout manifest (33 attack / 16 benign), per-config SHA-256",
     "generated once and frozen; episode seeds 1301-1303",
     "code/evaluation/final_safeagent_v3/gen_holdout_v3.py",
     "holdout_v3_freeze.json", "holdout", "1301,1302,1303",
     "Sec. 5 (RQ3), Tables 7-8, Fig. 4", MIT, "released"),
    ("code/configs/ijcip_final_v3/holdout_v3_freeze.json",
     "Freeze record for the holdout manifest (manifest SHA-256, seeds)",
     "generated at freeze time",
     "code/evaluation/final_safeagent_v3/gen_holdout_v3.py",
     "-", "holdout", "-", "Sec. 5 (RQ3)", MIT, "released"),
    ("code/configs/ijcip_final_safeagent_20260810/evidence_gate_test.csv",
     "Frozen 32-configuration Evidence-Gate test manifest",
     "frozen before one-shot gate evaluation",
     "code/evaluation/final_safeagent/gen_gate_sets.py",
     "evidence_gate_freeze.json", "test", "-",
     "Sec. 5 (RQ2), Table 5", MIT, "released"),
    ("code/configs/ijcip_final_safeagent_20260810/evidence_gate_dev_benign.csv",
     "Development benign configurations for gate calibration",
     "development split (never used for held-out reporting)",
     "code/evaluation/final_safeagent/gen_gate_sets.py",
     "-", "dev", "-", "Sec. 5 (RQ2)", MIT, "released"),
    ("code/configs/ijcip_final_safeagent_20260810/impact_estimator_holdout.csv",
     "Frozen 12-configuration estimator-validation holdout manifest",
     "frozen before estimator validation",
     "code/evaluation/final_safeagent/gen_holdout_configs.py",
     "impact_estimator_freeze.json", "holdout", "-",
     "Sec. 5 (RQ3), Table 9", MIT, "released"),
    ("code/configs/ijcip_final_safeagent_20260810/impact_estimator_dev.csv",
     "Estimator development manifest",
     "development split",
     "code/evaluation/final_safeagent/gen_holdout_configs.py",
     "-", "dev", "-", "Sec. 5 (RQ3)", MIT, "released"),
    ("code/configs/ijcip_final_safeagent_20260810/eh_duration_prior.json",
     "Frozen duration prior for the horizon-aware estimator (EH)",
     "fit on development data, frozen",
     "code/evaluation/final_safeagent/run_p0a_estimator.py",
     "impact_estimator_freeze.json", "dev", "-",
     "Sec. 5 (RQ3), Table 9", MIT, "released"),
    ("code/configs/ijcip_final_safeagent_20260810/opendss_llm_subset.csv",
     "Pre-registered 12-configuration OpenDSS AC power-flow subset",
     "pre-registered before the OpenDSS check",
     "code/evaluation/final_safeagent/run_p1a_opendss.py",
     "opendss_llm_subset_freeze.json", "test", "-",
     "Sec. 5 (RQ3), OpenDSS check", MIT, "released"),
    ("code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv",
     "25-configuration StubFeeder development scenario manifest",
     "development scenario library",
     "code/simulation/scenarios/ijcip_revision/generate_matrix.py",
     "scenario_matrix.yaml", "dev", "-",
     "Sec. 5 (frozen assets); Supplement S4", MIT, "released"),
    # ---- v3 canonical raw results -----------------------------------------
    ("code/results/ijcip_final_v3/holdout_e2e/holdout_e2e_raw.csv",
     "End-to-end holdout raw metrics: 8 arms x 49 configs x 3 episodes",
     "executed on the frozen holdout under strict LLM serving (0 fallbacks)",
     "code/evaluation/final_safeagent_v3/run_holdout_e2e.py",
     "holdout_v3_manifest.csv + evidence_gate_v3_frozen.json",
     "holdout", "1301,1302,1303",
     "Tables 7-8, Fig. 4", MIT, "released"),
    ("code/results/ijcip_final_v3/adversarial/adversarial_raw_qwen.csv",
     "Canonical 70-case adversarial suite raw rows, Qwen backbone, 8 arms",
     "real QLoRA inference, strict serving, sustained input",
     "code/evaluation/final_safeagent_v3/run_adversarial_v3.py --backend qwen",
     "evidence_gate_v3_frozen.json; suite seed 0, 5/family",
     "adversarial-test", "0", "Table 3, Fig. 2", MIT, "released"),
    ("code/results/ijcip_final_v3/adversarial/adversarial_raw_llama.csv",
     "Canonical 70-case adversarial suite raw rows, Llama backbone, 8 arms",
     "real QLoRA inference, strict serving, sustained input",
     "code/evaluation/final_safeagent_v3/run_adversarial_v3.py --backend llama",
     "evidence_gate_v3_frozen.json; suite seed 0, 5/family",
     "adversarial-test", "0", "Table 4", MIT, "released"),
    ("code/results/ijcip_final_v3/adversarial/adversarial_summary_qwen.csv",
     "Per-arm adversarial containment summary (Qwen)",
     "aggregated from adversarial_raw_qwen.csv",
     "code/evaluation/final_safeagent_v3/run_adversarial_v3.py",
     "-", "adversarial-test", "0", "Table 3, Fig. 2", MIT, "released"),
    ("code/results/ijcip_final_v3/adversarial/adversarial_summary_llama.csv",
     "Per-arm adversarial containment summary (Llama)",
     "aggregated from adversarial_raw_llama.csv",
     "code/evaluation/final_safeagent_v3/run_adversarial_v3.py",
     "-", "adversarial-test", "0", "Table 4", MIT, "released"),
    ("code/results/ijcip_final_v3/gate_robustness/gate_v3_summary.csv",
     "Evidence-Gate one-shot evaluation summary with Clopper-Pearson CIs",
     "frozen gate on frozen 32-config test set + FDI-magnitude probes",
     "code/evaluation/final_safeagent_v3/run_gate_robustness.py --evaluate",
     "evidence_gate_v3_frozen.json", "test", "-",
     "Table 5, Fig. 3", MIT, "released"),
    ("code/results/ijcip_final_v3/gate_robustness/gate_v3_raw.csv",
     "Evidence-Gate one-shot evaluation per-configuration rows",
     "frozen gate on frozen test set",
     "code/evaluation/final_safeagent_v3/run_gate_robustness.py --evaluate",
     "evidence_gate_v3_frozen.json", "test", "-",
     "Table 5", MIT, "released"),
    ("code/results/ijcip_final_v3/eh_robustness/eh_robustness.csv",
     "EH estimator robustness across 10 stress conditions on the 12-config holdout",
     "counterfactual estimator validation",
     "code/evaluation/final_safeagent_v3/run_eh_robustness.py",
     "impact_estimator_holdout.csv + eh_duration_prior.json",
     "holdout", "-", "Table 9", MIT, "released"),
    ("code/results/ijcip_final_v3/statistics/F1_physical.csv",
     "Primary paired configuration-level statistics (ENS/curtailment vs D0), Holm-adjusted",
     "paired sign-flip permutation + bootstrap; RNG seed 20260811",
     "code/evaluation/final_safeagent_v3/run_stats_v3.py",
     "-", "holdout", "20260811", "Table 8", MIT, "released"),
    ("code/results/ijcip_final_v3/statistics/F2_adversarial.csv",
     "Adversarial-family statistics pass-through",
     "derived from adversarial summaries",
     "code/evaluation/final_safeagent_v3/run_stats_v3.py",
     "-", "adversarial-test", "20260811", "Sec. 5 (RQ1)", MIT, "released"),
    ("code/results/ijcip_final_v3/statistics/F3_gate.csv",
     "Exact McNemar benign false-action comparison D0 vs OPROJ",
     "derived from holdout_e2e_raw.csv",
     "code/evaluation/final_safeagent_v3/run_stats_v3.py",
     "-", "holdout", "20260811", "Sec. 5 (RQ2)", MIT, "released"),
    ("code/results/ijcip_final_v3/deadline/latency_distributions.csv",
     "Deadline re-analysis latency distributions (v3 schema)",
     "re-analysis of archived strict-serving predictions",
     "code/evaluation/final_safeagent_v3/run_deadline_v3.py",
     "-", "-", "-", "Sec. 5 (RQ4)", MIT, "released"),
    ("code/results/ijcip_final_v3/deadline/slo_admission.csv",
     "SLO admission analysis (1-60 s replay)",
     "re-analysis of archived strict-serving predictions",
     "code/evaluation/final_safeagent_v3/run_deadline_v3.py",
     "-", "-", "-", "Sec. 5 (RQ4)", MIT, "released"),
    ("code/results/ijcip_final_v3/p0_forensics/p0a_result.json",
     "Pre-action gate invariance forensics (11,020 ticks, 0 violations)",
     "audit over archived holdout run traces",
     "code/evaluation/ijcip_final_v3/p0a_gate_forensics.py",
     "-", "holdout", "-", "Table 7 caption", MIT, "released"),
    ("code/results/ijcip_final_v3/p0_forensics/p0bcd_result.json",
     "Lineage assertions over archived runs (99/99 PASS)",
     "audit over archived run manifests",
     "code/evaluation/ijcip_final_v3/p0_forensics.py",
     "-", "-", "-", "Sec. 5", MIT, "released"),
    ("code/results/ijcip_final_v3/property_tests/property_test_result.json",
     "Structural enumeration (40,824 states, 0 violations) + 3 mutation guards",
     "exhaustive discretised decision-state enumeration",
     "code/evaluation/ijcip_final_v3/property_safety_tests.py",
     "-", "-", "-", "Sec. 5 (RQ1)", MIT, "released"),
    ("code/results/ijcip_final_v3/experiment_registry.json",
     "Registry of all v3 experiments with status and findings",
     "maintained during the final revision", "-", "-", "-", "-",
     "Sec. 5", MIT, "released"),
    ("code/results/ijcip_final_v3/environment.json",
     "Archived execution environment of the original v3 run (GPU, CUDA, package versions, adapter SHAs)",
     "recorded 2026-08-12 on the experiment machine", "-", "-", "-", "-",
     "Sec. 5; docs/REPRODUCIBILITY.md", MIT, "released"),
    ("code/results/ijcip_final_v3/artifact_manifest.json",
     "SHA-256 manifest recorded by the original v3 run",
     "recorded at run time", "-", "-", "-", "-", "-", MIT, "released"),
    # ---- supporting canonical results (older tags cited by the paper) ------
    ("code/results/ijcip_final/latency/latency_distributions.csv",
     "Canonical consultation-latency distribution (146 K=1 / 48 K=3 calls per backbone)",
     "summary of latency_ms in the four archived lora_eval prediction runs; "
     "re-derivable value-identically via scripts/derive_latency_v3.py",
     "scripts/derive_latency_v3.py",
     "-", "-", "-", "Table 10", MIT, "released"),
    ("code/results/ijcip_final/gate_auth/gate_auth_summary.csv",
     "Command-authentication sub-study summary (3 policies)",
     "hardcoded scenario set, seeds 2101-2103",
     "code/evaluation/final_safeagent_v3/run_gate_auth.py",
     "-", "-", "2101,2102,2103", "Table 6", MIT, "released"),
    ("code/results/ijcip_final/gate_auth/gate_auth_raw.csv",
     "Command-authentication sub-study raw rows",
     "hardcoded scenario set, seeds 2101-2103",
     "code/evaluation/final_safeagent_v3/run_gate_auth.py",
     "-", "-", "2101,2102,2103", "Table 6", MIT, "released"),
    ("code/results/ijcip_final/opendss_check/opendss_check_summary.csv",
     "Static IEEE-13/34 AC power-flow check summary (12 configs, 0 non-converged)",
     "OpenDSS power-flow snapshots on the pre-registered subset",
     "code/evaluation/final_safeagent_v3/run_opendss_check.py",
     "opendss_llm_subset.csv", "test", "0",
     "Sec. 5 (RQ3 physical-fidelity check)", MIT, "released"),
    ("code/results/ijcip_final/opendss_check/opendss_check_raw.csv",
     "Static OpenDSS power-flow check raw rows",
     "OpenDSS power-flow snapshots",
     "code/evaluation/final_safeagent_v3/run_opendss_check.py",
     "opendss_llm_subset.csv", "test", "0",
     "Sec. 5 (RQ3)", MIT, "released"),
    ("code/results/ijcip_final_revision/e6_adversarial/e6_raw.csv",
     "Unshielded-baseline adversarial raw rows (bare_qwen isolate 61/70) cited by the claim-consistency gate",
     "real QLoRA inference, strict serving",
     "code/evaluation/final_revision/run_e6_adversarial.py",
     "-", "adversarial-test", "-", "Sec. 5 (RQ1)", MIT, "released"),
    ("code/results/ijcip_final_safeagent_20260810/p1c_ablation/p1c_by_arm_qwen.csv",
     "Historical 42-case ablation per-arm metrics cited by the claim-consistency gate",
     "superseded by the 70-case suite; retained as claim-audit evidence",
     "code/evaluation/final_safeagent/run_p1c_ablation.py",
     "-", "adversarial-test", "-", "claim-consistency tests", MIT, "released"),
    ("code/results/ijcip_final_safeagent_20260810/p0b_gate/p0b_summary.csv",
     "Historical gate coverage summary cited by the claim-consistency gate",
     "superseded by the v3 gate evaluation; retained as claim-audit evidence",
     "code/evaluation/final_safeagent/run_p0b_gate.py",
     "-", "test", "-", "claim-consistency tests", MIT, "released"),
    # ---- fine-tuning corpus and adapters ----------------------------------
    ("code/finetuning_dataset/processed/train.jsonl",
     "Synthetic v0.1 supervised fine-tuning corpus, training split (161 examples)",
     "100% synthetic generator (no real-world data, no PII)",
     "code/finetuning_dataset/build_dataset.py --seed 0",
     "schema.py", "train", "0", "Sec. 4 (fine-tuning); availability statement",
     CC0, "released"),
    ("code/finetuning_dataset/processed/val.jsonl",
     "Synthetic corpus, validation split (21 examples)",
     "100% synthetic generator",
     "code/finetuning_dataset/build_dataset.py --seed 0",
     "schema.py", "val", "0", "Sec. 4", CC0, "released"),
    ("code/finetuning_dataset/processed/test.jsonl",
     "Synthetic corpus, test split (18 examples)",
     "100% synthetic generator",
     "code/finetuning_dataset/build_dataset.py --seed 0",
     "schema.py", "test", "0", "Sec. 4", CC0, "released"),
    ("code/finetuning_dataset/processed/manifest.json",
     "Per-row SHA-256 manifest of the corpus splits",
     "generated with the corpus",
     "code/finetuning_dataset/build_dataset.py --seed 0",
     "-", "-", "0", "availability statement", CC0, "released"),
    ("code/finetuning_dataset/revision_eval/eval_corpus.jsonl",
     "Leakage-controlled 146-prompt evaluation corpus",
     "synthetic, scenario-derived",
     "code/finetuning_dataset/revision_eval/build_eval_corpus.py",
     "-", "eval", "-", "Supplement S5; Table 10 provenance", CC0, "released"),
    ("code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter/adapter_model.safetensors",
     "QLoRA adapter for Qwen2.5-7B-Instruct (r=16, alpha=32, 3 epochs, seed 0); short SHA 0c9dec242e2f",
     "trained on the synthetic corpus; base weights NOT included",
     "code/finetuning/train.py --config configs/lora_qwen25_7b_local.yaml",
     "config_resolved.yaml", "train", "0",
     "Sec. 4-5; docs/MODEL_PROVENANCE.md",
     "adapter weights: released under the repository license; base model: Apache-2.0 (Qwen), obtain separately",
     "released (Git LFS)"),
    ("code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter/adapter_model.safetensors",
     "QLoRA adapter for Llama-3.1-8B-Instruct (r=16, alpha=32, 3 epochs, seed 0); short SHA 9a2b8436ce9c",
     "trained on the synthetic corpus; base weights NOT included",
     "code/finetuning/train.py --config configs/lora_llama31_8b_local.yaml",
     "config_resolved.yaml", "train", "0",
     "Sec. 4-5; docs/MODEL_PROVENANCE.md",
     "adapter weights: Llama 3.1 Community License terms apply to derivatives; base model: obtain separately",
     "released (Git LFS)"),
    # ---- mock fixture ------------------------------------------------------
    ("code/datasets/unsw_mg24/raw/alerts_mock.csv",
     "Synthetic mock alert fixture standing in for the non-redistributable UNSW-MG24 CSV",
     "RNG-generated mock (loader.synthesize_alerts, seed 0)",
     "code/datasets/unsw_mg24/loader.py",
     "-", "-", "0", "external-benchmark scaffolding", CC0, "released"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for tree in HASHED_TREES:
        for p in sorted((ROOT / tree).rglob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = sha256(p)
    (OUT_DIR / "sha256_all.json").write_text(json.dumps(
        {"schema": "sha256_all/v1",
         "note": "SHA-256 of every canonical data file in the release; "
                 "verify with scripts/verify_artifacts.py",
         "files": files}, indent=1) + "\n")
    print(f"[json] sha256_all.json ({len(files)} files)")

    arts = []
    for (rel, desc, src, gen, cfg, split, seed, ref, lic, redist) in CURATED:
        p = ROOT / rel
        if not p.exists():
            raise SystemExit(f"curated manifest references missing file: {rel}")
        arts.append({
            "artifact_id": Path(rel).name.replace(".", "_"),
            "relative_path": rel,
            "description": desc,
            "source": src,
            "generation_script": gen,
            "configuration": cfg,
            "split": split,
            "seed": seed,
            "sha256": sha256(p),
            "paper_reference": ref,
            "license": lic,
            "redistribution_status": redist,
        })
    # reference artifacts (manuscript-matching tables/figures)
    for p in sorted((ROOT / "artifacts/reference").rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(ROOT))
            arts.append({
                "artifact_id": "ref_" + p.name.replace(".", "_"),
                "relative_path": rel,
                "description": "Byte-exact reference copy of the manuscript "
                               "table/figure file for regeneration comparison",
                "source": "submitted manuscript build",
                "generation_script":
                    "code/evaluation/final_safeagent_v3/build_artifacts_v3.py "
                    "(except table_provenance/table_gate_auth: hand-authored, "
                    "see docs/PAPER_ARTIFACT_MAP.md)",
                "configuration": "-", "split": "-", "seed": "-",
                "sha256": sha256(p),
                "paper_reference": "Tables 2-10 / Figures 2-4",
                "license": MIT, "redistribution_status": "released",
            })
    (OUT_DIR / "release_manifest.json").write_text(json.dumps(
        {"schema": "release_manifest/v1",
         "system": "DER-SafeAgent (repository legacy name: DER-SecAgent)",
         "artifacts": arts}, indent=1) + "\n")
    print(f"[json] release_manifest.json ({len(arts)} curated artifacts)")


if __name__ == "__main__":
    main()
