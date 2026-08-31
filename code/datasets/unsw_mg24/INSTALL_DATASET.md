# Installing the UNSW-MG24 dataset

UNSW-MG24~\cite{Zhang2025_UNSW_MG24} is a heterogeneous cybersecurity dataset
for realistic microgrid systems. We do not redistribute it; obtain a copy
through the published research-data agreement and place a **CSV** at
`raw/alerts.csv` with at least these columns:

| Column        | Type    | Required | Notes                                           |
|---------------|---------|----------|-------------------------------------------------|
| `timestamp`   | float   | yes      | seconds since the run start; integer-bucketed   |
| `src_ip`      | string  | yes      | attacker / source IP                            |
| `dst_ip`      | string  | yes      | target IP (typically a microgrid asset)         |
| `protocol`    | string  | yes      | tcp / udp / modbus / dnp3 / iec61850            |
| `attack_type` | string  | yes      | DoS / Probe / R2L / U2R / MITM / BruteForce     |
| `src_port`    | int     | optional |                                                 |
| `dst_port`    | int     | optional |                                                 |

Any extra columns are passed through to the event payload and made available
to the agent prompt.

## Verifying installation
```bash
python3 -c "
from code.datasets.unsw_mg24 import UNSWMG24Loader, UNSWMG24Feeder
loader = UNSWMG24Loader('code/datasets/unsw_mg24/raw/alerts.csv')
print('alerts:', loader.total_alerts, '| horizon:', loader.time_horizon_s, 's')
feeder = UNSWMG24Feeder(loader=loader)
sample, events = feeder.read(180.0)
print('events at t=180:', len(events))
"
```

## Synthetic mock (no real data)
For CI-style smoke runs, a 50-alert mock can be generated without the real
dataset:
```bash
python3 -c "
from code.datasets.unsw_mg24.loader import synthesize_alerts
synthesize_alerts('code/datasets/unsw_mg24/raw/alerts_mock.csv', n_alerts=50, seed=0)
"
```

## Running through the harness
The loader exposes a `FeederAdapter` (`UNSWMG24Feeder`) that the same
`run_scenario()` consumes. Energy-impact metrics computed on this feeder are
not physically meaningful (no power flow); only detection and FP/FN
columns of `physical_metrics.csv` should be reported for UNSW-MG24 runs, as
discussed in paper §5.1.
