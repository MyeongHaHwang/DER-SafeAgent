# P0-B decision — 2026-08-10 (final code)

Frozen protocol: thresholds calibrated on dev (8 benign + 24 dev attacks as
stochastic streams), frozen (p_hard=5, soft path effectively disabled)
BEFORE the 32-configuration test set was run. Closed-loop runs use the final
architecture code (staleness signature + trigger); a pre-fix run set is
archived at runs_prestale_code/.

| System | Benign FA | FA norm/susp | Attack cov. | Attack ENS (kWh) | Benign LLM calls | p95 step (ms) |
|---|---|---|---|---|---|---|
| D0-G0 | 1.000 | 1.00/1.00 | 1.000 | 0.004 | 0 | 0.23 |
| D0-G1 | 0.188 | 0.00/0.375 | 1.000 | 0.306 | 0 | 0.28 |
| Q1-G0 | 0.000 | 0.00/0.00 | 0.438 | 4.19 | 36 | 0.13 |
| Q1-G1 | 0.000 | 0.00/0.00 | 0.562 | 4.25 | 6 | 0.28 |

**Decision: the Incident Evidence Gate is ADOPTED** in the final
architecture.

Findings:
1. Under genuine exogenous stochasticity the ungated deterministic pipeline
   false-acts on EVERY benign episode (16/16). The frozen gate removes all
   benign-normal false actions; the residual 3/16 are exactly the benign
   historian-echo configurations (protocol-level ambiguity by construction).
2. Attack coverage under D0 is unchanged (1.000). The measured cost is
   +0.30 kWh mean ENS on attacks, mostly DoS cases where the ungated
   pipeline had been "pre-mitigating" via benign-day false freezes - an
   artifact the gate exposes, not a capability it removes.
3. The gate reduces benign-day LLM consultations 36 -> 6 (advisory-path
   exposure to benign/attacker-observable content), and IMPROVES the model
   arm's attack coverage (0.438 -> 0.562): without the gate, a benign
   transient consumes the incident-level consultation and the stale benign
   hypothesis is cached into the attack window.
4. The advisory path alone (Q1) misses attacks the deterministic path
   catches (coverage 0.56 vs 1.00) and pays ~4 kWh ENS - consistent with
   the deployment posture in which the deterministic fast path responds in
   parallel and the model is advisory only. Reported honestly in the paper.
5. Unnecessary benign curtailment is ~0 kWh in all arms because the typical
   false action is freeze-at-nominal; the false-action cost in this
   environment is operational, not energetic.
6. Physical-only (soft) evidence could not separate attacks from benign PV
   transients at any calibrated setting - a negative sub-finding reported
   as such.
7. Added latency: FeatureView recomputation in the gate wrapper raises p95
   step time from ~0.23 ms to ~0.28 ms (negligible against control ticks).
