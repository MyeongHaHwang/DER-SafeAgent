"""Manifest and checksum integrity tests.

These are thin pytest wrappers around the release verification logic so CI
fails loudly on a corrupted or incomplete artifact set. The full check is
``python3 scripts/verify_artifacts.py`` (``make verify``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = ROOT / "data/manifests/release_manifest.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_release_manifest_well_formed():
    man = json.loads(RELEASE_MANIFEST.read_text())
    assert man["artifacts"], "empty release manifest"
    required = {"artifact_id", "relative_path", "description", "source",
                "sha256", "paper_reference", "license",
                "redistribution_status"}
    for e in man["artifacts"]:
        assert required <= set(e), f"malformed entry: {e.get('artifact_id')}"
        assert len(e["sha256"]) == 64


def test_canonical_raw_results_match_digests():
    """Spot-check the primary raw result files behind Tables 3-10."""
    man = json.loads(RELEASE_MANIFEST.read_text())
    idx = {e["relative_path"]: e["sha256"] for e in man["artifacts"]}
    primary = [
        "code/results/ijcip_final_v3/holdout_e2e/holdout_e2e_raw.csv",
        "code/results/ijcip_final_v3/adversarial/adversarial_raw_qwen.csv",
        "code/results/ijcip_final_v3/adversarial/adversarial_raw_llama.csv",
        "code/results/ijcip_final_v3/gate_robustness/gate_v3_summary.csv",
        "code/results/ijcip_final_v3/eh_robustness/eh_robustness.csv",
        "code/results/ijcip_final/latency/latency_distributions.csv",
    ]
    for rel in primary:
        assert rel in idx, f"{rel} missing from release manifest"
        assert _sha(ROOT / rel) == idx[rel], f"digest mismatch: {rel}"


def test_verify_artifacts_script_passes():
    """The full verification script must exit 0 on a pristine checkout.

    Adapter safetensors may be absent when Git LFS objects were not fetched;
    the script is invoked with --no-lfs here so the checksum layer of every
    non-LFS artifact is still enforced in that situation (missing LFS files
    are reported SKIPPED by the script, and are separately enforced by
    `make verify` on a full checkout).
    """
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_artifacts.py"), "--no-lfs"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"verify_artifacts failed:\n{r.stdout}\n{r.stderr}"


def test_no_machine_local_paths_outside_frozen_manifests():
    """The release must not leak machine-local paths.

    Exception (documented in docs/DATA_PROVENANCE.md): the hash-frozen
    configuration manifests and archived result snapshots record the absolute
    paths of the original experiment machine; rewriting them would invalidate
    their recorded SHA-256 digests, so they are kept byte-exact and remapped
    at load time by code.simulation.portable_paths.
    """
    allowed = {
        # frozen (hash-pinned) manifests — byte-exact by design
        "code/configs/ijcip_final_v3/holdout_v3_manifest.csv",
        "code/configs/ijcip_final_safeagent_20260810/evidence_gate_test.csv",
        "code/configs/ijcip_final_safeagent_20260810/evidence_gate_dev_benign.csv",
        "code/configs/ijcip_final_safeagent_20260810/impact_estimator_dev.csv",
        "code/configs/ijcip_final_safeagent_20260810/impact_estimator_holdout.csv",
        "code/configs/ijcip_final_safeagent_20260810/opendss_llm_subset.csv",
        "code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv",
    }
    hits = []
    SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                 "artifacts", "node_modules"}
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix in {".safetensors", ".png", ".pdf",
                                           ".bin", ".pt", ".pyc"}:
            continue
        if SKIP_DIRS & set(p.relative_to(ROOT).parts[:-1]):
            continue
        rel = str(p.relative_to(ROOT))
        if rel in allowed:
            continue
        needle = "/home/" + "mh-hwang"     # split so this file never matches itself
        try:
            if needle in p.read_text(errors="ignore"):
                hits.append(rel)
        except OSError:
            continue
    assert not hits, f"machine-local paths leaked in: {hits}"
