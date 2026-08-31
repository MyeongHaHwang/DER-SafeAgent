# Release Checklist — DER-SafeAgent public repository

Items marked `[x]` were completed **and verified** during release
preparation (evidence in `REPRODUCIBILITY_REPORT.md` and
`RELEASE_AUDIT.md`; last audit 2026-08-29). Items marked `[ ]` are open and
block the corresponding step.

## Publication blockers (require author action)

- [ ] **Push the release branch and merge to `main`.** The remote
      `main` (commit `23f0694…`, 38 files) holds only the documentation
      skeleton of the 2026-08-16 package plus the logo and a
      merge-conflicted LICENSE; the full verified tree lives on the local
      branch `reproducibility-release-v1`. Push it, open a PR, confirm CI
      is green, and merge — do **not** force-push.
- [ ] **Create the release tag `v1.1.0`** after the merge. The manuscript's
      Code and Data Availability section names `v1.1.0` as the version of
      record; the tag must exist before submission.
- [x] **Software license confirmed: Apache-2.0** (author decision,
      2026-08-30). LICENSE now carries the Apache-2.0 text with a scope
      note for third-party assets; CITATION.cff, pyproject.toml, README,
      and every release-manifest entry declare Apache-2.0 consistently
      (the initial upload's merge-conflicted LICENSE is resolved). The
      Llama QLoRA adapter remains additionally subject to the Llama 3.1
      Community License; the synthetic corpus remains CC0-1.0.
- [ ] **Confirm author metadata.** The author block (order, corresponding
      author, email raphael9290@gmail.com) was restored from the group's
      prior DER-SecAgent submission; confirm it, and supply the
      author-only items the repository cannot know: ORCID iDs, funding
      statement, CRediT roles, conflict-of-interest statement,
      Declaration of generative-AI use in writing, Acknowledgements
      (tracked in `paper/IJCIP_AUTHOR_CHECKLIST.md` of the internal tree;
      deliberately not invented here).
- [ ] **Git LFS on the hosting side.** `.gitattributes` tracks
      `*.safetensors` via LFS (two adapters, 154/161 MB). After the first
      push run a fresh `git clone && git lfs pull && make verify` and
      confirm the adapter digests check out.
- [ ] **CI first run.** `.github/workflows/ci.yml` (verification ladder +
      hygiene + LaTeX build job) is untested until the branch is on GitHub;
      confirm the first run is green (checkout must fetch LFS).
- [ ] **Docker build.** `docker build -t der-safeagent . && docker run
      --rm der-safeagent` — unverified locally (no docker daemon); run once
      anywhere with Docker.
- [ ] **Archival DOI (optional).** No Zenodo/archival deposit exists and
      none is cited. If desired: GitHub release → Zenodo → add the DOI to
      the availability section and CITATION.cff.

## Completed and verified (2026-08-29 audit)

- [x] Claim-by-claim manuscript audit — all 35 principal claims traced to
      raw sources and recomputed (`docs/CLAIM_LEDGER.md`); zero mismatches.
- [x] Table 9 ModelOnly CI discrepancy resolved by re-analysis with shared
      resampling draws (no raw data touched; CHANGELOG 1.1.0).
- [x] Table 8 closed-loop-coverage metric traced and documented
      (`docs/CLAIM_LEDGER.md`, `docs/EXPERIMENTS.md`; pre-action gate
      invariance 11,020 ticks / 0 violations).
- [x] Action policy single source of truth — YAML export + manuscript
      Table 3 generated from `safety_projection.py`, guarded by
      `tests/test_action_policy.py`.
- [x] Causal-language audit — arm independence stated in §5.2; ModelOnly
      labelled diagnostic; "always escalate" corrected to
      vetoed-or-escalated everywhere it appears.
- [x] Third-party license review — UNSW-MG24 (8.4 GB), TON_IoT (69 MB),
      copyrighted PDFs, and CICFlowMeter binary **excluded**; base-model
      weights never included; Llama-derivative status of the Llama adapter
      stated in `LICENSE` and `docs/MODEL_PROVENANCE.md`.
- [x] Secret scan passed — `make hygiene`: no credentials, tokens, keys, or
      private endpoints; machine-local path prefixes only inside the 7
      hash-frozen manifests (documented exception, enforced by
      `tests/test_manifests.py` and `scripts/repo_hygiene.py`).
- [x] Large-file review passed — ~315 MB of the ~360 MB tree is the two
      LFS adapters; largest non-LFS file ≈16 MB (adapter tokenizer) plus
      the 6 MB logo; nothing over GitHub's 100 MB limit as a normal blob.
- [x] Clean-environment verification — fresh copy + fresh venv,
      CPU-only, `make release-check` end-to-end (see RELEASE_AUDIT.md).
- [x] Artifact hashes verified — `make verify`: 236 checks, 0 failures
      (release manifest 63 curated artifacts; full listing 153 files).
- [x] Manuscript values reproduced — 8 generated tables and 3 figure PNGs
      byte-identical; statistics hash-identical; latency value-identical.
- [x] Code and Data Availability matches the release — cites the verified
      repository URL (`github.com/MyeongHaHwang/DER-SafeAgent`) and the
      `v1.1.0` version of record; no invented DOI.
- [x] Stale anonymization wording removed (IJCIP review is
      single-anonymized); real author list in the manuscript and
      CITATION.cff.
- [x] Final manuscript PDF built — 52 pp from the tracked `paper/` source;
      availability section appears once; no undefined references.
- [x] GPU serving path spot-verified on the reference machine (2026-08-16)
      — released Qwen adapter loads under strict mode and serves a real
      call; recorded `adapter_sha` matches every canonical run manifest.

## Recommended (non-blocking)

- [ ] Enable branch protection + require the CI check on `main`.
- [ ] Attach the two adapters to the GitHub release as assets for non-LFS
      users (digests in `docs/MODEL_PROVENANCE.md`).
- [ ] After acceptance: add the camera-ready citation (volume/DOI) to
      `CITATION.cff` `preferred-citation`.
