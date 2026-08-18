# Release Checklist — DER-SafeAgent public repository

Items marked `[x]` were completed **and verified** during release
preparation (2026-08-16; evidence in `REPRODUCIBILITY_REPORT.md`). Items
marked `[ ]` are open and block the corresponding step.

## Publication blockers (require author action)

- [ ] **Verified public repository URL.** No git remote or hosting account
      is configured on the preparation machine, so no URL exists yet and
      none was invented. After the authors create the repository and push,
      insert the URL in exactly two places:
      1. `paper/sections/8-availability.tex` — replace the clause
         "it has been prepared as a public repository, and the repository
         URL (with an archival DOI) will be added upon acceptance" with the
         real URL (and DOI), then rebuild with `tectonic main.tex`;
      2. this repository's `README.md` Contact section and `CITATION.cff`
         (`repository-code:` field).
- [ ] **Archival DOI, if applicable.** No Zenodo/archival deposit exists;
      none was cited. If desired: create the GitHub release → archive to
      Zenodo → add the DOI alongside the URL (same two places).
- [ ] **Author-approved software license.** MIT was inherited from the
      internal tree and kept, with a scope carve-out added for third-party
      models/datasets. The authors must confirm MIT (and the carve-out
      wording) before publishing.
- [ ] **De-anonymisation timing.** Publishing this repository during
      double-blind review is an author/venue decision. Note that the
      hash-frozen manifests unavoidably contain the original machine's
      filesystem prefix (including a username) — kept byte-exact because
      rewriting them would invalidate the recorded protocol digests (see
      `docs/DATA_PROVENANCE.md`). If the repository must go public *before*
      acceptance, this identity leak must be accepted or the freeze
      digests regenerated (a protocol-provenance trade-off only the
      authors can approve). At camera-ready, also fill author names in
      `CITATION.cff`, `LICENSE`, and the README Contact section.
- [ ] **Git LFS on the hosting side.** Before the first push run
      `git lfs install && git lfs track` is already configured via
      `.gitattributes`; confirm the two adapter `.safetensors` (154/161 MB)
      upload as LFS objects, then run `make verify` on a fresh clone.
- [ ] **CI first run.** `.github/workflows/ci.yml` is untested until the
      repository exists on GitHub; confirm the first run is green
      (checkout must fetch LFS).
- [ ] **Docker build.** `docker build -t der-safeagent . && docker run --rm
      der-safeagent` — unverified locally (no docker daemon); run once
      anywhere with Docker.

## Completed and verified

- [x] Third-party license review — UNSW-MG24 (8.4 GB), TON_IoT (69 MB),
      copyrighted PDFs, and CICFlowMeter binary **excluded**; base-model
      weights never included; Llama-derivative status of the Llama adapter
      stated in `LICENSE` and `docs/MODEL_PROVENANCE.md`; MIT scope
      carve-out added.
- [x] Secret scan passed — no credentials, tokens, keys, private
      endpoints, or vendor API usage anywhere (design excludes vendor
      SDKs). Machine-local path prefixes removed everywhere except the
      hash-frozen manifests (documented exception, enforced by
      `tests/test_manifests.py`).
- [x] Large-file review passed — repository is ~350 MB total, of which
      ~315 MB is the two LFS adapters; largest non-LFS file ≈16 MB
      (adapter tokenizer); no file exceeds GitHub's 100 MB limit as a
      normal blob.
- [x] Clean-environment tests passed — fresh venv, CPU-only:
      `make test` (172 passed), `make smoke`, `make verify` (227 checks),
      `make reproduce-paper`, plus re-execution of gate-evaluate /
      eh-robustness / deadline / gate-auth / holdout-cpu / property-tests
      reproducing canonical outputs (byte- or value-identical; see report).
- [x] Artifact hashes verified — `make verify`: release manifest (60
      curated artifacts), full digest listing (147 files), freeze-JSON
      pins, and the original run's `artifact_manifest.json`; 0 failures.
- [x] Manuscript values reproduced — all 7 generated main-text tables and
      all 3 figure PNGs byte-identical to the submitted manuscript's
      files; statistics hash-identical; latency summary value-identical.
- [x] Code and Data Availability statement matches the release — rewritten
      as a single section describing exactly what this package contains
      (no invented URL/DOI; truthful pre-release wording).
- [x] Final manuscript PDF built successfully — `tectonic main.tex`,
      49 pages, availability section appears once, no undefined
      references, no placeholders, no new overfull boxes.
- [x] GPU serving path spot-verified on the reference machine — released
      Qwen adapter loads under strict mode and serves a real call;
      recorded `adapter_sha` matches every canonical run manifest.

## Recommended (non-blocking)

- [ ] Enable branch protection + require the CI check on `main`.
- [ ] Add a GitHub release `v1.0.0` mirroring `CHANGELOG.md`, attaching
      the two adapters as release assets for non-LFS users (digests in
      `docs/MODEL_PROVENANCE.md`).
- [ ] After acceptance: add the camera-ready citation (volume/DOI) to
      `CITATION.cff` `preferred-citation`.
