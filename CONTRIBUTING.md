# Contributing

This repository is primarily a **reproducibility package** for the
DER-SafeAgent manuscript, so its first duty is to keep the released code,
frozen protocols, and canonical results verifiable. Contributions are
welcome within that constraint.

## Ground rules

1. **Never modify canonical artifacts.** The frozen configuration manifests
   (`code/configs/**`), the archived canonical results (`code/results/**`),
   and the manuscript reference artifacts (`artifacts/reference/**`) are
   checksum-pinned. `make verify` must pass on every commit. If an
   experiment is legitimately re-run and replaces canonical data, say so
   explicitly in the PR and regenerate the manifests with
   `scripts/make_release_manifests.py` — never regenerate manifests to make
   a failing check pass.
2. **Safety invariants are the contract.** Any change touching
   `code/Multi_AI_Agent/` or `code/simulation/` must keep the full test
   suite green, in particular `test_runtime_safe_gate.py`,
   `test_safety_projection.py`, and
   `code/evaluation/final_safeagent_v3/test_metric_invariants.py`.
3. **No fabricated model output.** LLM-labelled results must be produced
   under strict serving (`DER_LLM_STRICT=1`); heuristic fallbacks are for
   tests only and are recorded as such in every trace.
4. **No third-party data or weights in the tree.** See `data/README.md` for
   how external datasets are acquired; base-model weights are never
   committed.

## Workflow

```bash
bash scripts/bootstrap.sh      # venv + core deps + smoke + verify
make test                      # full suite
make reproduce-paper           # must still match the manuscript byte-for-byte
```

CI runs the same ladder (see `.github/workflows/ci.yml`). Please keep new
code consistent with the existing style: standard library + numpy/pandas,
`pathlib.Path` for paths, repository-root-relative paths, module drivers
runnable as `python3 -m code.…` from the repository root.
