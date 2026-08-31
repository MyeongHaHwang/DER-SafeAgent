# Manuscript source

LaTeX source of the manuscript
*DER-SafeAgent: A Runtime-Assurance Architecture for Safe LLM-Assisted
Cyber-Physical Incident Response in Distributed Energy Resources*
(submitted to the International Journal of Critical Infrastructure
Protection) and its supplementary document.

Build (any engine works; the release was verified with tectonic, an
XeTeX-based engine that fetches `elsarticle` automatically):

```bash
cd paper
tectonic main.tex            # manuscript
tectonic supplementary.tex   # supplementary material (S1-S10)
```

The numeric tables in `tables/final_v3/` and the figures in
`figure/final_v3/` are generated files: `make reproduce-paper` at the
repository root regenerates them from the canonical result files and
compares them byte-for-byte against `artifacts/reference/`.
`tables/final_v3/table_action_policy.tex` regenerates with
`python3 scripts/export_action_policy.py`. Do not edit generated tables by
hand — edit the generators.

The compiled PDF is not tracked; build it locally.
