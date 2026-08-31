# Multi_AI_Agent — runtime-assurance agent stack

Implementation of the DER-SafeAgent runtime path: the runtime-safe
`FeatureView` (`telemetry_features.py`), the Evidence Gate
(`evidence_gate.py`), the safety-projection shield (`safety_projection.py`),
the class-aware avoidance map (`attack_class_map.py`), the energy/horizon
estimators (`energy_estimator.py`, `horizon_estimator.py`), the fixed
mitigation registry and coordinator, the hash-chained audit record
(`audit_chain.py`), and the LLM adapter (`adapter.py`).

The package name retains the legacy system name (`DER-SecAgent` →
`Multi_AI_Agent`); renaming it would invalidate the configuration hashes
recorded in the frozen manifests. See the repository `README.md` for the
naming note.

Prompts live in `prompts/` (SHA-256-pinned in run manifests); synthetic
worked examples live in `examples/` — all identifiers there (IP addresses,
asset names, Modbus/DNP3 fields) are synthetic and refer to IEEE 13/34-bus
test-feeder nodes, not real equipment.

Safety-invariant tests: `test_runtime_safe_gate.py` (ground-truth flags
cannot influence the runtime FeatureView), `test_safety_projection.py`
(irreversible-action escalation, no-op veto rule, registry-only execution).
