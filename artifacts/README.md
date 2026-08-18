# artifacts/ — manuscript artifact regeneration

| Directory | Contents | Tracked in git? |
|---|---|---|
| `reference/tables/` | Byte-exact copies of the 9 table files included in the submitted manuscript (7 generated + 2 hand-authored; see `docs/PAPER_ARTIFACT_MAP.md`) | yes (checksummed) |
| `reference/figures/` | Byte-exact copies of the manuscript's Figures 2–4 (PDF + 300-dpi PNG) | yes (checksummed) |
| `tables/`, `figures/` | Output of `make reproduce-paper` — regenerated from the canonical raw results | no (regenerated) |
| `processed/` | Small derived CSVs emitted during regeneration (consultation counts, optional latency re-derivation) | no (regenerated) |
| `raw/` | Reserved for outputs of local re-execution; the canonical raw results themselves live in `code/results/` (their pipeline-native, hash-recorded location) | no |

Regenerate and compare:

```bash
make reproduce-paper
```

Verified for this release: all 7 generated tables and all 3 figure PNGs are
byte-identical to `reference/`; figure PDFs match modulo embedded timestamp
metadata; the statistics CSVs regenerate hash-identically (fixed RNG seed
20260811).
