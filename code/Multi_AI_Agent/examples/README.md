# Worked examples — alert → mitigation traces

Two end-to-end traces showing how the five-agent loop converts a raw
ICS-protocol alert into a mitigation primitive. These are the artefacts
referenced from §4 of the paper (R2.C4) and are checked-in verbatim so
the reproduction is byte-stable.

| Example | Protocol | Attack class | Final action |
|---------|----------|--------------|--------------|
| `01_modbus_command_spoof.json` | Modbus/TCP (FC=06 single-register write) | `command_spoof` | `freeze_setpoint` |
| `02_dnp3_fdi_inverter.json`    | DNP3 outstation analog input report (g30v5) | `fdi`           | `freeze_setpoint` |

Each file contains:

- `raw_alert`: the IDS-side packet summary fed to the harness.
- `harness_event`: the canonical `EventLog` we hand to the detector.
- `feature_view`: the deterministic FeatureView emitted by the
  Telemetry Analyst.
- `hypothesis`: the K=3 self-consistency aggregate from the Hypothesis
  Agent.
- `impact_estimates`: the per-candidate $(E_{\text{curt}}, \mathrm{ENS},
  \text{tier})$ triples from the Energy Impact Agent.
- `caution`: the Caution Agent's verdict.
- `coordinator`: the final action and reason.
- `mitigation_script`: the substation-side OT command that the
  StackStorm adapter would execute (terminal-ready). For Modbus/DNP3
  the script is an SCL/CIP-style `iec61850-cli` invocation; the format
  is documented in `docs/mitigation_dispatch.md`.

The numeric fields below were captured from a real run of the v2 pipeline
on a synthetic harness step; they match the public-API outputs of
`code/Multi_AI_Agent/adapter.py::DERSecAgentDetector.step` exactly.
