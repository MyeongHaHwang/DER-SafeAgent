# Data Card — DER-SecAgent Fine-tuning Corpus v0.1

## Overview
Supervised fine-tuning corpus for LLM agents performing **DER cybersecurity event triage**.
Each example is a multi-turn conversation: the agent receives event/telemetry context and must
produce a structured detection + recommended mitigation grounded in DER operational impact.

- **Format**: JSONL, one example per line, conforming to `schema.py::Example`.
- **Target task**: classify attack class, locate affected DER asset, propose mitigation,
  estimate expected energy impact (qualitative tier).
- **Conversation style**: ChatML / OpenAI messages format (`role`, `content`).

## Measured statistics — v0.1 (`processed/manifest.json`, seed=0)

| Quantity | Value |
|---|---|
| Total examples | **200** |
| Total characters | **109\,632** |
| Estimated tokens (4 chars / token) | **27\,408** |
| Train / val / test rows | **161 / 21 / 18** |
| Distinct sources | 1 (synthetic shard) |
| Distinct scenarios | 3 |
| Distinct attack classes | 6 |
| Per-role chars (sys / user / assistant) | 26\,800 / 35\,835 / 46\,997 |

These numbers are reproduced exactly by `python build_dataset.py --seed 0` from the
checked-in `raw/synthetic/shard_dummy.json` shard. SHA-256s of the emitted splits:

| Split | SHA-256 |
|---|---|
| `train.jsonl` | `9de44c2780c9a8ab2baffa8517faab4ac9350797b9aa36734730deb3257b4845` |
| `val.jsonl`   | `c5eca360474238a9105182d84e5a447b05714da1360eed66b789dc4fa9ceeda4` |
| `test.jsonl`  | `e7dd89ba15595ad4e567eeaf6d1d0d4142d38d2ed4b061b01d7b6ae9557e59af` |

### Attack-class distribution (across all splits)

| Class | Count |
|---|---:|
| `none`           | 41 |
| `command_spoof`  | 35 |
| `dos`            | 34 |
| `fdi`            | 31 |
| `firmware`       | 30 |
| `replay`         | 29 |

### Scenario distribution (across all splits)

| Scenario | Count |
|---|---:|
| `ieee34_command_spoof_derms` | 74 |
| `ieee13_fdi_inverter`        | 72 |
| `ieee13_replay_dnp3`         | 54 |

## Source mix (target proportions for v1.0; v0.1 is 100% synthetic)

| Source | v0.1 share | v1.0 target | Notes |
|--------|-----------:|------------:|-------|
| Synthetic from `code/simulation/` runs | 100% | 60% | Programmatically generated event logs + ground-truth labels |
| Public CTI / advisories (CISA ICS-CERT, MITRE ATT&CK for ICS) | 0% | 20% | Paraphrased into triage Q&A |
| Re-purposed open SCADA/DER logs (e.g., HAI, SWaT, DERMS testbed traces) | 0% | 15% | License-checked subsets only |
| Human-curated edge cases | 0% | 5% | Hand-written by team for hard negatives |

## Splits
Stratified by `(scenario, attack_class)`:
- train: 80% (measured: 161/200 = 80.5%)
- val: 10% (measured: 21/200 = 10.5%)
- test: 10% (measured: 18/200 = 9.0%, held out — no leakage from synthetic pipeline seed)

## Schema (summary, see `schema.py`)
```
Example {
  id: str
  scenario: str          # e.g. "ieee13_fdi_inverter"
  attack_class: str      # one of {none, fdi, replay, command_spoof, dos, firmware}
  messages: [             # ChatML
    {role: "system",    content: ...},
    {role: "user",      content: <event log + telemetry snapshot>},
    {role: "assistant", content: <structured JSON: detection + action + impact_tier>}
  ]
  metadata: {
    source: "synthetic|cti|public_log|human"
    license: SPDX id
    seed: int | null
    generated_at: ISO-8601
  }
}
```

## Licensing & ethics
- Each example carries an SPDX license tag in metadata.
- No PII; synthetic IPs/identities only.
- Public-log subsets are filtered against original dataset licenses; rejected examples
  are listed in `raw/REJECTED.md` with reason.

## Versioning
- v0.1 (released): 200 examples (measured), 100% synthetic, for pilot fine-tuning runs.
- v1.0 (planned): ~50k examples after simulation campaign in WS2.
- Each version freezes a `manifest.json` with row hashes and split assignment.

## Known risks
- Synthetic bias toward modeled scenarios (mitigated: hold-out scenarios in test).
- Label noise from auto-generation (mitigated: rule + LLM dual-pass labeling, disagreement → human).
- Distribution shift vs. real utility data (acknowledged limitation, see paper §6).
