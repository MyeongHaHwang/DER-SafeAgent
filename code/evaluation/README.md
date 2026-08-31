# Evaluation

## Subfolders
- `energy_metrics/` — ENS, voltage/frequency, ramp, curtailment computations.
- `stats/` — paired bootstrap CI, multi-seed aggregation, multiple-comparison correction.

## Entry point (planned)
`run_all.py`:
1. Iterate scenarios × methods × seeds.
2. Compute detection metrics + energy-impact metrics.
3. Emit `results/summary.csv` + per-metric figures into `paper/figure/`.
