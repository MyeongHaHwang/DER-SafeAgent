# Release Audit — DER-SafeAgent v1.1.0 release candidate

- **Audit date:** 2026-08-29
- **Audited tree:** local branch `reproducibility-release-v1`
  (parent: remote `main` @ `23f069497480f83422264027d6cec0a2530abfb7`;
  the exact release-candidate commit IDs are in `git log`)
- **Operating system:** Linux 6.17.0-29-generic (Ubuntu), x86_64
- **Python:** 3.12.3
- **CPU/GPU:** CPU-only verification for this audit (the GPU rows of
  `REPRODUCIBILITY_REPORT.md` were verified 2026-08-16 on the reference
  RTX 3080 machine and were not re-run)
- **Clean-environment dependency set** (fresh virtualenv,
  `pip install -r requirements.txt` only): numpy 2.5.2, pandas 3.0.5,
  scipy 1.18.1, matplotlib 3.10.9, PyYAML 6.0.3, pytest 9.1.1
- **Tracked-candidate file count:** ~688 files (excluding `__pycache__`,
  build outputs, and per-run dumps), ~360 MB, of which ~315 MB is the two
  QLoRA adapters (`*.safetensors`, Git LFS) and 6 MB the logo PNG
- **Git LFS files:** `code/finetuning/results/20260519-144102-lora_qwen25_7b_local/adapter/adapter_model.safetensors`
  (154 MB, sha256 `0c9dec24…`),
  `code/finetuning/results/20260519-144624-lora_llama31_8b_local/adapter/adapter_model.safetensors`
  (161 MB, sha256 `9a2b8436…`)

## Verification results (all executed on the audit date)

| Check | Command | Result |
|---|---|---|
| Test suite | `make test` | **PASS** — 179 passed, 0 skipped (manuscript claim-consistency checks active against the tracked `paper/` source) |
| Smoke test | `make smoke` | **PASS** — all 4 stages |
| Manifest verification | `make verify` (and `--strict`) | **PASS** — 236 checks, 0 failed, 1 skip (untracked compiled `paper/main.pdf`) |
| Paper-artifact reproduction | `make reproduce-paper` | **PASS** — 14 comparisons: 8 tables byte-identical, 3 figure PNGs byte-identical, 3 statistics CSVs digest-identical |
| Latency re-derivation | `make derive-latency` | **PASS** — value-identical from the 4 archived `predictions.jsonl` runs |
| Repository hygiene | `make hygiene` | **PASS** — 0 findings (secrets, stale anonymization wording, oversize non-LFS files, absolute paths outside the 7 documented frozen CSVs) |
| LaTeX build | `make latex` (tectonic 0.15.0) | **PASS** — `main.pdf` 52 pages, `supplementary.pdf` builds, 0 undefined references/citations |
| Clean-copy end-to-end | fresh copy + fresh virtualenv + `make release-check` | **PASS** — full gate green with only `requirements.txt` installed |
| Docker image build | `docker build …` | **NOT RUN** — no docker daemon on the audit machine (open item in RELEASE_CHECKLIST.md) |
| GitHub Actions | `.github/workflows/ci.yml` | **NOT RUN** — executes on first push (open item) |
| GPU/LLM re-execution | `make holdout-llm`, `make adversarial` | **NOT RE-RUN** — canonical raw results shipped + checksummed; see REPRODUCIBILITY_REPORT.md |
| Git clone/LFS round-trip | `git clone && git lfs pull && make verify` | **NOT RUN** — git/git-lfs are not installed on the audit machine; the branch was constructed with a pure-Python git implementation (dulwich) on top of the fetched remote history, with the adapters committed as LFS pointers and their objects staged under `.git/lfs/objects/`. Must be verified after the first push (open item) |

## Notable audit findings and resolutions

1. **Remote `main` was a 38-file skeleton.** The public repository
   contained only the documentation/scripts of the 2026-08-16 package —
   no `code/`, no `data/manifests/`, no reference artifacts, no adapters —
   so `make verify`/`make reproduce-paper` could not succeed from a clone,
   while the README claimed they did. This release restores the complete
   verified tree on `reproducibility-release-v1` (no history rewrite; the
   remote commit is the branch's parent).
2. **Remote `LICENSE` contained unresolved merge-conflict markers**
   (`<<<<<<< HEAD` MIT+carve-out vs the GitHub-template Apache-2.0).
   Resolved to **Apache-2.0** (author-confirmed, 2026-08-30) with the
   third-party scope note retained; all package license declarations
   (CITATION.cff, pyproject, README, release manifests) updated
   consistently. The remote-added logo and README logo line were
   preserved.
3. **Table 9 ModelOnly CI inconsistency** (identical paired vectors,
   different CIs) — root-caused to sequential RNG stream consumption;
   fixed by shared resampling draws (common random numbers); manuscript
   and canonical derived statistics updated; no raw data touched.
4. **Groundedness-audit source data was cited but not shipped** — the
   `e7_trustworthiness` CSVs behind the 70–73%/23–30% figures are now
   released and manifest-pinned.
5. **Stale double-blind wording** in README/CITATION/LICENSE/SECURITY
   removed (IJCIP review is single-anonymized); real author metadata
   restored from the group's prior submission (author-confirmation items
   listed in RELEASE_CHECKLIST.md).
6. **Manuscript page count** is 52 with the de-anonymized author block,
   the new action-policy Table 3, and the Related-Work additions
   (previous anonymized build: 49; table numbering 3–10 → 4–11 is
   synchronized across all repository documentation).

## Audit-machine reliability note

During the audit the preparation machine intermittently (a) returned
corrupted buffers on single large (>100 MB) file reads — caught because the
LFS staging step verifies every adapter against its canonical SHA-256 with
independent chunked re-reads, and (b) produced sporadic interpreter
segfaults at unrelated code points, which disappeared on retry while the
identical commands passed repeatedly before and after. All shipped
artifacts were verified with chunked hashing (the method `make verify`
uses) and match their recorded digests. A single-invocation
`make release-check` passed end-to-end on this content (and on the
clean-copy environment); as the instability worsened, every stage of the
gate (deps, compileall, 179 tests, smoke, verify, reproduce-paper,
derive-latency, hygiene, LaTeX 52 pp) was additionally re-run to PASS as
separate processes on the exact release tree. The authors are advised to
run a memory test on the preparation machine, re-run `make release-check`
once on a healthy machine or in CI, and re-run `make verify` on a fresh
clone after pushing (already checklist items).

## Remaining blockers

See "Publication blockers" in `RELEASE_CHECKLIST.md`: push + CI first run,
`v1.1.0` tag, author metadata (ORCID/funding/CRediT/COI/GenAI
declaration), post-push LFS round-trip check, Docker build. The license is
confirmed (Apache-2.0).
