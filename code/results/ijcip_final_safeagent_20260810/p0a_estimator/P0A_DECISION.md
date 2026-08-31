# P0-A decision — 2026-08-10

Frozen protocol: dev = 25 legacy configurations (informed the design);
holdout = 12 new causally-different configurations, manifest + duration prior
SHA-frozen before evaluation; counterfactual rollouts never consult any
estimator; top-1 uses ε-tie handling (regret ≤ 0.01 kWh-eq ⇒ optimal).

| Estimator | split | top-1(ε) | top-1 strict | Kendall τ | median regret | mean regret | p95 regret | zero-regret frac | curt MAE | ENS MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| E60 | dev | 0.32 | 0.00 | 0.53 | 90.32 | — | 191.2 | 0.32 | 28.4 | 15.0 |
| EH  | dev | 1.00 | 0.96 | 1.00 | 0.00 | 0.00 | 0.0 | 1.00 | 22.9 | 15.6 |
| EMH | dev | 1.00 | 0.96 | 1.00 | 0.00 | 0.00 | 0.0 | 1.00 | 22.9 | 15.6 |
| E60 | holdout | 0.33 [CI 0.08,0.58] | 0.00 | 0.51 | 14.91 | 76.34 | 226.2 | 0.33 | 31.6 | 13.8 |
| EH  | holdout | **1.00 [CI 1.0,1.0]** | 1.00 | 1.00 | 0.00 | 0.00 | 0.0 | 1.00 | 26.3 | 14.7 |
| EMH | holdout | 1.00 [CI 1.0,1.0] | 1.00 | 1.00 | 0.00 | 0.00 | 0.0 | 1.00 | 26.3 | 14.7 |

**Decision: CASE A — EH is retained** in the final DER-SafeAgent architecture
as a deterministic component. E60 is retired (horizon-mis-calibrated by
design). EMH is numerically identical to EH on both splits (cost terms are
linear in the horizon, so marginalising the duration prior equals using its
mean residual); EMH is therefore NOT kept as a separate component — reported
as a design observation.

**What the repair actually is.** Three decision-time ingredients, all
deterministic: (i) a class-family hypothesis from protocol-level evidence
(forged commands = spoof-like; tamper flags = FDI-like; duplicated/stale
telemetry = replay-or-DoS, which are indistinguishable at decision time
because DoS freezes the whole sample); (ii) integration over a dev-calibrated
mean-residual incident horizon instead of a fixed 60 s window; (iii)
restoration-horizon accounting for absorbing (irreversible) actions. The
stale-family ambiguity is physically harmless: freeze_setpoint is optimal
under DoS and near-zero-regret under replay — a minimax property the 60 s
estimator could not see.

**Honest limitations, to be carried into the paper.**
- Magnitude accuracy remains poor (curtailment MAE ~26 kWh on holdout): EH
  ranks actions correctly but does not predict impact magnitudes precisely.
  No claim of accurate impact *prediction* may be made.
- Perfect holdout ranking reflects the structured StubFeeder physics; the
  same construction on a real feeder would inherit real model error.
- The candidate registry is small (5 actions); ranking 5 actions is an easier
  problem than a production DERMS action space.
