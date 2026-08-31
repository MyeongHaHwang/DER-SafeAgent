# Scenarios

Each scenario is a folder containing:
- `config.yaml` — feeder, DER assets, attack params, duration.
- `feeder.dss` (or GridLAB-D `.glm`).
- `attack.py` — injection hook.
- `expected.json` — labels for evaluation.

Initial scenario list:
1. `ieee13_pv_bess_baseline` — no attack (control).
2. `ieee13_fdi_inverter` — false data injection on PV inverter telemetry.
3. `ieee34_replay_dnp3` — replay attack on DNP3.
4. `ieee34_command_spoof_derms` — spoofed DERMS dispatch command.
