# Energy-Impact Metrics

Functions to implement (one per file):
- `ens.py` — Energy Not Supplied integration.
- `voltage.py` — node-level deviation stats.
- `frequency.py` — frequency excursion metrics.
- `ramp.py` — IEEE 1547 ramp violation count.
- `curtailment.py` — PV/BESS curtailed energy.

Each takes `timeseries.csv` from `code/simulation/runs/...` and returns a dict.
