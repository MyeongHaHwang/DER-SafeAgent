#!/usr/bin/env python3
"""Verify the integrity of every released artifact against SHA-256 manifests.

Checks, in order:
  1. ``data/manifests/release_manifest.json`` — the curated manifest of every
     canonical configuration, dataset, adapter, raw result, and reference
     artifact: file exists and SHA-256 matches. Malformed manifest entries are
     errors.
  2. ``data/manifests/sha256_all.json`` — the full-tree digest listing taken
     at packaging time: every listed file exists and matches.
  3. Internal freeze-JSON pins — the frozen protocol files record the digests
     of the manifests they froze; those cross-checks must hold.
  4. ``code/results/ijcip_final_v3/artifact_manifest.json`` — the digests
     recorded by the original experiment run; entries whose files are not
     part of this release (``paper/main.pdf``) are reported SKIPPED with the
     reason, never silently passed.
  5. ``--strict`` additionally fails if unexpected files are present in the
     frozen configuration directories.

Exit code is non-zero if any check fails. Adapter weight files are stored via
Git LFS; a missing adapter is a FAILURE unless ``--no-lfs`` is given, in which
case it is reported SKIPPED (lfs not fetched).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_MANIFEST = ROOT / "data/manifests/release_manifest.json"
FULL_MANIFEST = ROOT / "data/manifests/sha256_all.json"
V3_MANIFEST = ROOT / "code/results/ijcip_final_v3/artifact_manifest.json"

REQUIRED_FIELDS = {"artifact_id", "relative_path", "description", "source",
                   "sha256", "paper_reference", "license",
                   "redistribution_status"}

FREEZE_PINS = [
    # (freeze json, key holding the digest, pinned file)
    ("code/configs/ijcip_final_v3/holdout_v3_freeze.json", "manifest_sha256",
     "code/configs/ijcip_final_v3/holdout_v3_manifest.csv"),
    ("code/configs/ijcip_final_safeagent_20260810/evidence_gate_freeze.json",
     "test_manifest_sha256",
     "code/configs/ijcip_final_safeagent_20260810/evidence_gate_test.csv"),
    ("code/configs/ijcip_final_safeagent_20260810/impact_estimator_freeze.json",
     "manifest_sha256",
     "code/configs/ijcip_final_safeagent_20260810/impact_estimator_holdout.csv"),
    ("code/configs/ijcip_final_safeagent_20260810/impact_estimator_freeze.json",
     "prior_sha256",
     "code/configs/ijcip_final_safeagent_20260810/eh_duration_prior.json"),
    ("code/configs/ijcip_final_safeagent_20260810/opendss_llm_subset_freeze.json",
     "manifest_sha256",
     "code/configs/ijcip_final_safeagent_20260810/opendss_llm_subset.csv"),
]

# artifact_manifest.json entries not shipped in the public release
NOT_SHIPPED = {
    "paper/main.pdf": "manuscript PDF is distributed by the journal, not this repository",
}

LFS_SUFFIXES = {".safetensors"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_lfs_pointer(p: Path) -> bool:
    try:
        head = p.open("rb").read(120)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="also fail on unexpected files in frozen config dirs")
    ap.add_argument("--no-lfs", action="store_true",
                    help="report missing/pointer LFS files as SKIPPED, not FAIL")
    args = ap.parse_args()

    failures: list[str] = []
    skipped: list[str] = []
    n_ok = 0

    def check_file(rel: str, want: str, origin: str) -> None:
        nonlocal n_ok
        p = ROOT / rel
        if not p.exists() or looks_like_lfs_pointer(p):
            if p.suffix in LFS_SUFFIXES and args.no_lfs:
                skipped.append(f"{rel} (git-lfs object not fetched; --no-lfs)")
                return
            failures.append(f"[{origin}] missing file: {rel}")
            return
        got = sha256(p)
        if got != want:
            failures.append(f"[{origin}] checksum mismatch: {rel}\n"
                            f"    expected {want}\n    got      {got}")
        else:
            n_ok += 1

    # 1. curated release manifest -------------------------------------------
    if not RELEASE_MANIFEST.exists():
        failures.append(f"missing manifest: {RELEASE_MANIFEST}")
    else:
        try:
            man = json.loads(RELEASE_MANIFEST.read_text())
            entries = man["artifacts"]
        except (json.JSONDecodeError, KeyError) as e:
            failures.append(f"malformed release manifest: {e}")
            entries = []
        for i, e in enumerate(entries):
            missing = REQUIRED_FIELDS - set(e)
            if missing:
                failures.append(f"malformed manifest entry #{i} "
                                f"({e.get('artifact_id', '?')}): missing "
                                f"fields {sorted(missing)}")
                continue
            check_file(e["relative_path"], e["sha256"], "release_manifest")

    # 2. full-tree digest listing -------------------------------------------
    if not FULL_MANIFEST.exists():
        failures.append(f"missing manifest: {FULL_MANIFEST}")
    else:
        allman = json.loads(FULL_MANIFEST.read_text())
        for rel, want in sorted(allman["files"].items()):
            check_file(rel, want, "sha256_all")

    # 3. freeze-JSON internal pins ------------------------------------------
    for fz, key, pinned in FREEZE_PINS:
        fzp = ROOT / fz
        if not fzp.exists():
            failures.append(f"missing freeze file: {fz}")
            continue
        want = json.loads(fzp.read_text()).get(key)
        if not want:
            failures.append(f"freeze file {fz} lacks key {key}")
            continue
        check_file(pinned, want, f"freeze:{Path(fz).name}")

    # 4. original experiment-run manifest -----------------------------------
    if V3_MANIFEST.exists():
        v3 = json.loads(V3_MANIFEST.read_text())
        for rel, want in sorted(v3["artifacts"].items()):
            if rel in NOT_SHIPPED:
                skipped.append(f"{rel} ({NOT_SHIPPED[rel]})")
                continue
            check_file(rel, want, "ijcip_final_v3/artifact_manifest")
    else:
        failures.append(f"missing manifest: {V3_MANIFEST}")

    # 5. strict mode: unexpected files in frozen config dirs ----------------
    if args.strict:
        allman = json.loads(FULL_MANIFEST.read_text()) if FULL_MANIFEST.exists() else {"files": {}}
        listed = set(allman["files"])
        for d in ("code/configs",):
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file():
                    rel = str(p.relative_to(ROOT))
                    if rel not in listed:
                        failures.append(f"[strict] unexpected file not in "
                                        f"sha256_all.json: {rel}")

    print(f"{n_ok} artifact checks passed, {len(skipped)} skipped, "
          f"{len(failures)} failed")
    for s in skipped:
        print(f"[SKIPPED] {s}")
    for f in failures:
        print(f"[FAIL] {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
