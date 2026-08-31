# Installing the official EPRI IEEE 13-bus / 34-bus feeders

The simplified `feeder.dss` shipped with each scenario is a small
implementation that exercises the harness end-to-end without requiring an
external download. For results reported in the paper, swap it with the
**official EPRI IEEE 13-bus and 34-bus testcases**.

## Why the official cases are not bundled
The EPRI OpenDSS distribution is permissively licensed but redistributing the
`IEEETestCases/` tree inside this repo creates avoidable license-bookkeeping
overhead. Instead, fetch them once from the upstream OpenDSS distribution.

## How to install
1. Install the EPRI OpenDSS package (Windows or Linux). The Python wheel
   `OpenDSSDirect.py` (already pinned in `requirements.txt`) ships its own
   solver but does *not* include the testcases.
2. Download the `IEEETestCases/` directory from the official EPRI repository:
   ```
   https://sourceforge.net/p/electricdss/code/HEAD/tree/trunk/Distrib/IEEETestCases/
   ```
   Mirror copies are also available in the EPRI OpenDSS Wiki and several
   academic repositories.
3. Replace each scenario's `feeder.dss` with the appropriate official file:

| Scenario | Replace with |
|----------|--------------|
| `ieee13_fdi_inverter/feeder.dss` | `IEEETestCases/13Bus/IEEE13Nodeckt.dss` (+ DER lines below) |
| `ieee13_command_spoof/feeder.dss` | same as above |
| `ieee34_*/feeder.dss` (planned) | `IEEETestCases/34Bus/Master.dss` (+ DER lines) |

4. Append the DER asset definitions referenced by `config.yaml`. Example for
   the 13-bus testcase (place at the end, before `Set voltagebases` /
   `CalcVoltageBases`):
   ```
   New Generator.INV_634 phases=3 bus1=634 kV=4.16 kW=140 kVA=200 model=1 conn=wye
   New Generator.BESS_675 phases=3 bus1=675 kV=4.16 kW=100 kVA=150 model=1 conn=wye
   ```

5. Verify the swap converged correctly:
   ```bash
   python3 -c "
   from code.simulation.feeder import OpenDSSFeeder
   import yaml
   from pathlib import Path
   cfg = yaml.safe_load(open('code/simulation/scenarios/ieee13_fdi_inverter/config.yaml'))
   f = OpenDSSFeeder(dss_path='code/simulation/scenarios/ieee13_fdi_inverter/feeder.dss',
                    monitored_buses=cfg['monitored_buses'], ders=cfg['ders'])
   sample, _ = f.read(0.0)
   print('voltages OK:', all(0.9 < v < 1.1 for v in sample.bus_voltages_pu.values()))
   "
   ```

## Provenance recording
Once installed, run the helper below to write a `feeder_provenance.json` next
to the replaced `feeder.dss` so reviewers can verify the source:

```bash
python3 code/simulation/scenarios/_record_provenance.py \
    code/simulation/scenarios/ieee13_fdi_inverter/feeder.dss
```

## Reproducibility notes
- The official feeders are deterministic; `feeder_provenance.json` plus the
  scenario `config.yaml` (containing attack windows and seed) is sufficient
  to re-derive every reported number.
- If you cannot download the official feeders (e.g., air-gapped environment),
  the bundled simplified feeder is a faithful proxy for the harness; the
  paper notes this explicitly in §5.1.
