# DER Co-Simulation

Default simulator: **OpenDSS** via `py-dss-interface` or `OpenDSSDirect.py`.
Optional: GridLAB-D for multi-domain checks.

## Layout
- `scenarios/` — one folder per scenario (feeder + attack injection).
  - `ieee13_pv_bess_baseline/`
  - `ieee13_fdi_inverter/`
  - `ieee34_replay_dnp3/`
  - `ieee34_command_spoof_derms/`
- `harness.py` — co-sim driver (sim ↔ agent ↔ SOAR).
- `attack_injectors/` — plug-in modules per technique.

## Outputs
Each run writes `runs/<scenario>/<seed>/timeseries.csv` consumed by `code/evaluation/`.
