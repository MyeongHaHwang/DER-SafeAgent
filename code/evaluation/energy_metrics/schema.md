# Time-series schema (input to energy_metrics)

All metric functions consume a `pandas.DataFrame` with the columns below.
Extra columns are ignored. The simulator (`code/simulation/harness.py`) is the
authoritative emitter and must produce a CSV that loads to this schema.

| Column | Unit | Required | Notes |
|--------|------|----------|-------|
| `t` | s | yes | monotonic, can be irregular |
| `load_demand` | kW | yes (for ENS) | feeder-head total demand |
| `load_served` | kW | yes (for ENS) | actually served |
| `v_pu_<bus>` | pu | yes (for voltage) | one column per monitored bus |
| `freq` | Hz | yes (for frequency) | single column for system frequency |
| `p_pv_<id>` | kW | yes (for ramp / curtailment) | dispatched PV active power |
| `p_pv_avail_<id>` | kW | yes (for curtailment) | available PV (no curtailment) |
| `p_bess_<id>` | kW | yes (for ramp) | + discharge / − charge |

## Conventions
- Bus / asset IDs use the simulator's native names (no normalization).
- Sampling rate is scenario-specific; functions assume regular spacing for
  durations and use trapezoidal integration for energies.
- Missing columns raise `ValueError` rather than returning silently zero.
