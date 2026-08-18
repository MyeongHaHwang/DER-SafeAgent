# Threat Model (summary)

The full threat model and the AI-pipeline threat taxonomy used in the paper
ship with the code documentation:

- [`code/docs/threat_model.md`](../code/docs/threat_model.md) — assets and
  protocol surfaces (smart inverters over IEC 61850 MMS, BESS over
  DNP3/Modbus, RTU/IED, DERMS/EMS), attacker capabilities, and the in/out of
  scope boundary (physical tampering, GPS spoofing, IT-only ransomware, and
  firmware supply-chain compromise are out of scope).
- [`code/docs/ai_pipeline_threat_taxonomy.md`](../code/docs/ai_pipeline_threat_taxonomy.md)
  — threats to the LLM pipeline itself (prompt injection via telemetry,
  adversarial evidence shaping, model-output-as-attack-vector), which the
  70-case adversarial suite operationalises.

## The trust boundary in one paragraph

The LLM (local QLoRA-served Qwen2.5-7B / Llama-3.1-8B) is an **untrusted
advisory component**: its output is treated exactly like other
attacker-influencible input. Deterministic mechanisms enforce containment
regardless of what the model says: (1) the **Evidence Gate** opens only on
sustained, observable, authentication-aware evidence — never on model
output; (2) the **safety-projection shield** maps any proposal into the
fixed five-action registry and forbids class-inappropriate actions;
(3) **irreversible actions** (`isolate_inverter`) proposed by the model
always escalate to a human; (4) model output may gate **neither**
irreversible action **nor** stand-down inaction (the symmetric veto rule —
a model `no_op` cannot suppress a response to independently evidenced
incidents); (5) the deterministic fast path survives model timeout/failure;
(6) every decision lands in a hash-chained, tamper-evident audit record.

These are not aspirational statements: each is enforced by tests
(`code/Multi_AI_Agent/test_safety_projection.py`,
`test_runtime_safe_gate.py`,
`code/evaluation/trustworthy_validation/test_trustworthy_properties.py`) and
by the structural enumeration (40,824 discretised decision states, 0
violations; three guard-removal mutations each caught by hundreds to
thousands of violating states —
`code/results/ijcip_final_v3/property_tests/property_test_result.json`).

The paper's conclusion, restated: **containment comes from the
deterministic runtime-assurance mechanisms, not from the model.** The
evaluation measures how much advisory value an LLM can add *inside* that
boundary — it does not claim the LLM improves physical outcomes.

## Research-testbed scope

All attack material targets the synthetic StubFeeder/OpenDSS research
testbed (IEEE 13/34-bus). See `SECURITY.md` for the responsible-use
statement.
