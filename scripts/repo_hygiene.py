#!/usr/bin/env python3
"""Repository hygiene check for public release (CPU, no network).

Fails (non-zero exit) on:
  * secret-shaped strings (AWS keys, GitHub/HF tokens, private keys);
  * machine-local absolute paths (/home/, /Users/, C:\\) outside the
    hash-frozen manifests whose bytes are pinned by recorded digests;
  * stale anonymization wording (double-blind / double-anonymized) in
    release documentation;
  * placeholder markers (TODO/FIXME/PLACEHOLDER/yourusername) in release
    documentation;
  * tracked-candidate files larger than 100 MB that are not covered by a
    Git LFS pattern in .gitattributes;
  * common junk (checkpoint dirs, __pycache__ is allowed locally but
    ignored, .env files, id_rsa).

Run from the repository root: python3 scripts/repo_hygiene.py
"""
from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The frozen protocol manifests keep their original bytes (their SHA-256 is
# what the experiment records pin), and they contain the original machine's
# absolute scenario paths. Consumers remap via code/simulation/portable_paths.
# This exception is documented in docs/DATA_PROVENANCE.md and enforced
# narrowly here and in tests/test_manifests.py.
FROZEN_PATH_EXCEPTIONS = [
    "code/configs/ijcip_final_safeagent_20260810/evidence_gate_dev_benign.csv",
    "code/configs/ijcip_final_safeagent_20260810/evidence_gate_test.csv",
    "code/configs/ijcip_final_safeagent_20260810/impact_estimator_dev.csv",
    "code/configs/ijcip_final_safeagent_20260810/impact_estimator_holdout.csv",
    "code/configs/ijcip_final_safeagent_20260810/opendss_llm_subset.csv",
    "code/configs/ijcip_final_v3/holdout_v3_manifest.csv",
    "code/configs/ijcip_revision_r1r2_20260805/scenario_manifest.csv",
]

SECRET_PATTERNS = [
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(rb"ghp_[A-Za-z0-9]{36}"), "GitHub token"),
    (re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(rb"hf_[A-Za-z0-9]{34}"), "Hugging Face token"),
    (re.compile(rb"-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----"),
     "private key"),
    (re.compile(rb"sk-[A-Za-z0-9]{32,}"), "API secret key"),
]

DOC_GLOBS = ["*.md", "LICENSE", "CITATION.cff", "SECURITY.md"]
STALE_WORDS = re.compile(r"double.blind|double.anonym", re.IGNORECASE)
PLACEHOLDERS = re.compile(r"\bTODO\b|\bFIXME\b|PLACEHOLDER|yourusername|<URL>")
ABS_PATH = re.compile(rb"/home/[a-z0-9_-]+/|/Users/[A-Za-z0-9_-]+/|C:\\\\Users")

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
BINARY_SUFFIXES = {".safetensors", ".png", ".pdf", ".pyc", ".zip", ".gz",
                   ".parquet", ".npz", ".pt", ".bin"}
# Documentation is allowed to *mention* the legacy paths when explaining the
# frozen-manifest exception itself.
DOC_PATH_MENTION_OK = {"docs/DATA_PROVENANCE.md", "RELEASE_CHECKLIST.md",
                       "RELEASE_AUDIT.md", "REPRODUCIBILITY_REPORT.md",
                       "docs/CLAIM_LEDGER.md"}


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_dir():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def lfs_patterns() -> list[str]:
    ga = ROOT / ".gitattributes"
    pats = []
    if ga.exists():
        for line in ga.read_text().splitlines():
            if "filter=lfs" in line:
                pats.append(line.split()[0])
    return pats


def main() -> int:
    failures: list[str] = []
    lfs = lfs_patterns()
    n = 0
    for p in iter_files():
        rel = str(p.relative_to(ROOT))
        n += 1
        data = p.read_bytes()

        for pat, label in SECRET_PATTERNS:
            if pat.search(data):
                failures.append(f"{label} pattern in {rel}")

        if p.suffix not in BINARY_SUFFIXES and rel not in FROZEN_PATH_EXCEPTIONS:
            if rel not in DOC_PATH_MENTION_OK and ABS_PATH.search(data):
                failures.append(f"machine-local absolute path in {rel}")

        if p.stat().st_size > 100 * (1 << 20):
            if not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat)
                       for pat in lfs):
                failures.append(f">100MB file not covered by Git LFS: {rel}")

        if p.name in {".env", "id_rsa", "id_ed25519", "credentials"}:
            failures.append(f"credential-shaped filename: {rel}")

    # History documents may legitimately *describe* the removal of the old
    # anonymization wording; only live documentation is checked for it.
    history_ok = {"CHANGELOG.md", "RELEASE_AUDIT.md"}
    for g in DOC_GLOBS:
        for p in list(ROOT.glob(g)) + list((ROOT / "docs").glob(g)):
            text = p.read_text(errors="replace")
            rel = str(p.relative_to(ROOT))
            if rel not in history_ok and STALE_WORDS.search(text):
                failures.append(f"stale double-blind wording in {rel}")
            if PLACEHOLDERS.search(text):
                failures.append(f"placeholder marker in {rel}")

    print(f"hygiene: scanned {n} files; {len(failures)} finding(s)")
    for f in failures:
        print(f"[FAIL] {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
