# Security Policy

## Scope and responsible use

DER-SafeAgent is a **research artifact**. Everything in this repository —
telemetry, alerts, attack scenarios, protocol traces, asset identifiers, and
IP addresses — is **synthetic or simulation-derived** (StubFeeder episodes
and static OpenDSS power-flow snapshots on IEEE 13/34-bus test feeders). No
operational utility data, credentials, endpoints, or real infrastructure
targets are included, and none were used during the research.

The adversarial material (attack injectors, the 70-case adversarial suite,
prompt-perturbation families) exists solely to evaluate the containment
properties of the runtime-assurance architecture on the simulated research
testbed. It does not provide capability against real DER installations, and
must not be pointed at systems you do not own or lack authorization to test.

This code is **not** an operational protection system. The paper's claims
are about containment of an untrusted advisory component inside a simulated
testbed; do not deploy any part of this repository to control real power
equipment.

## Reporting a vulnerability

If you find a security-relevant defect (e.g. a way for model output or
attacker-controlled telemetry to bypass the Evidence Gate, the
safety-projection shield, the irreversible-action escalation, or the audit
hash chain), please open a GitHub issue marked `security`, or contact the
maintainer listed in the README's Contact section. Reports that include a
failing test case in the style of
`code/Multi_AI_Agent/test_runtime_safe_gate.py` are especially welcome —
the safety-invariant test suite is the contract this project maintains.

## No secrets

The repository is designed to operate without any credentials: no vendor LLM
APIs are used (models are served locally), and no network services are
contacted at runtime. If you believe a secret or personal datum has been
committed, please report it as above.
