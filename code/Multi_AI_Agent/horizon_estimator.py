"""Horizon-aware Energy Impact estimators (P0-A, tag ijcip_final_safeagent_20260810).

The legacy estimator (``energy_estimator.py``, conceptually **E60**) integrates
candidate-action costs over a fixed 60 s response window. Counterfactual
validation (E4) showed that at the *incident* horizon the candidates differ by
two orders of magnitude more than at 60 s, and E60 never selected the optimal
action (top-1 0/25, median regret 90.3 kWh-eq). E60 is preserved unchanged as
the historical baseline; this module implements the repair candidates:

``EH``  — incident-horizon-aware. Estimates the *remaining* incident duration
          from decision-time observables only (class hypothesis from the
          deterministic FeatureView, elapsed anomaly duration, a
          dev-calibrated class-conditional duration prior via mean residual
          life), then integrates predicted deviation flows over that horizon.
``EMH`` — multi-horizon variant. Marginalises the same per-horizon cost over
          the dev-calibrated duration distribution instead of using the mean
          residual point estimate. Because most cost terms are linear in the
          horizon, EMH is expected to track EH except through the nonlinear
          ENS-threshold term; both are reported.

Decision-time information ONLY. The true attack end time is never an input;
the duration prior is calibrated on DEVELOPMENT configurations and frozen
before held-out evaluation.

Class hypotheses (deterministic, from the FeatureView's dominant signal —
never from the LLM):

- ``spoof_like``   forged command events, or a large physical deviation in
                   telemetry that carries no tamper flags: the asset's actual
                   output is believed wrong ⇒ deviation accrues until the
                   incident ends or a setpoint action lands.
- ``fdi_like``     tampered telemetry: the *reported* value is wrong but the
                   physical state is believed healthy ⇒ doing nothing is
                   physically free.
- ``stale``        duplicated/frozen telemetry (replay OR DoS — shown to be
                   indistinguishable at the telemetry level at decision time
                   because DoS freezes the whole sample): the physical state is
                   unknown; costs are the mean over the two mechanisms
                   (replay: healthy; DoS: target forced to zero).
- ``none``         no evidence ⇒ no_op.

The estimator knows the feeder's static ratings and the harness's published
physical accounting rules (nominal dispatch = 0.7 × rating; ENS accrues when
the fleet injection deficit exceeds 40 % of nominal) — a defender legitimately
has the grid model. It does NOT know the exogenous future.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .energy_estimator import ImpactEstimate, _tier_from_curt

CANDIDATES = ("no_op", "freeze_setpoint", "throttle_ramp",
              "request_ied_revalidation", "isolate_inverter")

ENS_WEIGHT = 5.0            # same scalarisation the counterfactual oracle uses
DEFICIT_ENS_THRESHOLD = 0.4  # harness: ENS accrues above 40 % fleet deficit
ENS_SERVE_DROP = 0.5         # harness: 50 % of demand unserved above threshold


@dataclass
class HorizonCalibration:
    """Dev-calibrated class-conditional incident-duration prior (seconds)."""
    durations_s: dict[str, list[float]] = field(default_factory=dict)
    floor_s: float = 30.0
    cap_s: float = 600.0

    @classmethod
    def from_durations(cls, per_class: dict[str, list[float]]) -> "HorizonCalibration":
        return cls(durations_s={k: sorted(float(x) for x in v)
                                for k, v in per_class.items()})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"durations_s": self.durations_s, "floor_s": self.floor_s,
             "cap_s": self.cap_s}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HorizonCalibration":
        d = json.loads(Path(path).read_text())
        return cls(durations_s=d["durations_s"], floor_s=d["floor_s"],
                   cap_s=d["cap_s"])

    def _pool(self, family: str) -> list[float]:
        pool = self.durations_s.get(family) or [
            x for v in self.durations_s.values() for x in v]
        return pool or [120.0]

    def mean_residual_s(self, family: str, elapsed_s: float) -> float:
        """E[d - elapsed | d > elapsed] under the dev empirical distribution."""
        pool = [d for d in self._pool(family) if d > elapsed_s]
        if not pool:
            return self.floor_s
        mr = sum(d - elapsed_s for d in pool) / len(pool)
        return min(max(mr, self.floor_s), self.cap_s)

    def residual_distribution(self, family: str,
                              elapsed_s: float) -> list[tuple[float, float]]:
        """[(remaining_s, weight)] for EMH marginalisation."""
        pool = [d - elapsed_s for d in self._pool(family) if d > elapsed_s]
        if not pool:
            return [(self.floor_s, 1.0)]
        w = 1.0 / len(pool)
        return [(min(max(r, self.floor_s), self.cap_s), w) for r in pool]


def class_family(fv) -> str:
    """Deterministic decision-time class hypothesis from the FeatureView."""
    g = (lambda k, d=0: getattr(fv, k, fv.get(k, d) if isinstance(fv, dict) else d))
    if g("n_command_events"):
        return "spoof_like"
    # FDI family: the legacy `n_tampered_events` oracle OR the runtime-safe
    # observable integrity residual (reported-power anomaly without a voltage
    # response) both indicate telemetry-integrity compromise with a healthy
    # asset, for which doing nothing is physically free.
    if g("n_tampered_events") or g("integrity_residual"):
        return "fdi_like"
    if (g("persistent_freeze") or g("n_dup_events") >= 2
            or g("telemetry_stale_ticks") >= 3 or g("seq_regressions") >= 2):
        return "stale"
    if g("dominant_signal", "none") == "telemetry_dev":
        return "spoof_like"     # physical deviation without tamper flags
    return "none"


def _asset_state(fv, target: str) -> tuple[float, float, float, float]:
    """(cur_kw, avail_kw, fleet_avail_kw, fleet_cur_kw) from the FeatureView."""
    p = fv.asset_p_kw if not isinstance(fv, dict) else fv.get("asset_p_kw", {})
    av = (fv.asset_p_avail_kw if not isinstance(fv, dict)
          else fv.get("asset_p_avail_kw", {}))
    cur = float(p.get(target, 0.0))
    avail = float(av.get(target, 0.0))
    return cur, avail, float(sum(av.values())), float(sum(p.values()))


def _flow_costs(family: str, cur: float, avail: float, fleet_avail: float,
                fleet_cur: float, load_kw: float | None) -> dict[str, dict]:
    """Per-action (curtail_kw, ens_kw) deviation FLOWS while the incident is
    active, plus one-off constants. Returned per action as
    {flow_curt_kw, flow_ens_kw, const_kwh, absorbing}. ``absorbing`` actions
    accrue their flow over the restoration horizon, not the incident horizon.
    """
    nominal_fleet = 0.7 * fleet_avail
    nominal_asset = 0.7 * avail

    def ens_flow(injection_kw: float) -> float:
        """kW of unserved demand while fleet injection is ``injection_kw``."""
        if load_kw is None or nominal_fleet <= 0:
            return 0.0
        deficit_frac = max(0.0, (nominal_fleet - injection_kw) / nominal_fleet)
        return ENS_SERVE_DROP * load_kw if deficit_frac > DEFICIT_ENS_THRESHOLD else 0.0

    # Believed *physical* output of the target under each hypothesis.
    if family == "spoof_like":
        phys = cur                      # telemetry truthful; output forced low
    elif family == "fdi_like":
        phys = nominal_asset            # reported value wrong, asset healthy
    elif family == "stale":
        phys = None                     # unknown: mix replay(healthy)/dos(zero)
    else:
        phys = nominal_asset

    fleet_rest = max(0.0, 0.7 * (fleet_avail - avail))  # other assets at nominal

    def per_mechanism(phys_kw: float) -> dict[str, dict]:
        dev = max(0.0, nominal_asset - phys_kw)         # deviation flow, kW
        inj_passive = fleet_rest + phys_kw              # if we do nothing
        inj_fixed = fleet_rest + nominal_asset          # if a setpoint action lands
        inj_isolated = fleet_rest                       # if we cut the asset
        return {
            "no_op": dict(flow_curt_kw=dev, flow_ens_kw=ens_flow(inj_passive),
                          const_kwh=0.0, absorbing=False),
            "request_ied_revalidation": dict(flow_curt_kw=dev,
                                             flow_ens_kw=ens_flow(inj_passive),
                                             const_kwh=0.02, absorbing=False),
            # freeze wins the setpoint race against the injector on the next
            # tick, so the deviation flow stops; small transition constant.
            "freeze_setpoint": dict(flow_curt_kw=0.0, flow_ens_kw=0.0,
                                    const_kwh=0.05 + dev / 3600.0,
                                    absorbing=False),
            "throttle_ramp": dict(flow_curt_kw=0.5 * dev,
                                  flow_ens_kw=ens_flow(0.5 * (inj_passive + inj_fixed)),
                                  const_kwh=0.1, absorbing=False),
            "isolate_inverter": dict(flow_curt_kw=nominal_asset,
                                     flow_ens_kw=ens_flow(inj_isolated),
                                     const_kwh=0.0, absorbing=True),
        }

    if phys is not None:
        return per_mechanism(phys)
    healthy, zero = per_mechanism(nominal_asset), per_mechanism(0.0)
    return {a: dict(flow_curt_kw=0.5 * (healthy[a]["flow_curt_kw"] + zero[a]["flow_curt_kw"]),
                    flow_ens_kw=0.5 * (healthy[a]["flow_ens_kw"] + zero[a]["flow_ens_kw"]),
                    const_kwh=0.5 * (healthy[a]["const_kwh"] + zero[a]["const_kwh"]),
                    absorbing=healthy[a]["absorbing"])
            for a in CANDIDATES}


def _integrate(flows: dict[str, dict], horizon_s: float,
               t_restore_s: float) -> list[ImpactEstimate]:
    out = []
    for a in CANDIDATES:
        f = flows[a]
        h = t_restore_s if f["absorbing"] else horizon_s
        curt = f["flow_curt_kw"] * h / 3600.0 + f["const_kwh"]
        ens = f["flow_ens_kw"] * h / 3600.0
        out.append(ImpactEstimate(
            action=a, expected_curt_kwh=curt, expected_ens_kwh=ens,
            tier=_tier_from_curt(curt + 2.0 * ens),
            rationale=f"horizon-aware: {'restoration' if f['absorbing'] else 'incident'}"
                      f" horizon {h:.0f}s"))
    return out


def estimate_eh(fv, target_asset: str, elapsed_s: float,
                calib: HorizonCalibration, load_demand_kw: float | None = None,
                t_restore_s: float = 600.0) -> list[ImpactEstimate]:
    """EH: point-estimate remaining horizon via mean residual life."""
    family = class_family(fv)
    cur, avail, fleet_av, fleet_cur = _asset_state(fv, target_asset)
    flows = _flow_costs(family, cur, avail, fleet_av, fleet_cur, load_demand_kw)
    h = calib.mean_residual_s(family, elapsed_s)
    return _integrate(flows, h, t_restore_s)


def estimate_emh(fv, target_asset: str, elapsed_s: float,
                 calib: HorizonCalibration, load_demand_kw: float | None = None,
                 t_restore_s: float = 600.0) -> list[ImpactEstimate]:
    """EMH: marginalise the per-horizon cost over the dev duration prior."""
    family = class_family(fv)
    cur, avail, fleet_av, fleet_cur = _asset_state(fv, target_asset)
    flows = _flow_costs(family, cur, avail, fleet_av, fleet_cur, load_demand_kw)
    mix: dict[str, dict[str, float]] = {a: {"curt": 0.0, "ens": 0.0}
                                        for a in CANDIDATES}
    for h, w in calib.residual_distribution(family, elapsed_s):
        for e in _integrate(flows, h, t_restore_s):
            mix[e.action]["curt"] += w * e.expected_curt_kwh
            mix[e.action]["ens"] += w * e.expected_ens_kwh
    return [ImpactEstimate(action=a, expected_curt_kwh=mix[a]["curt"],
                           expected_ens_kwh=mix[a]["ens"],
                           tier=_tier_from_curt(mix[a]["curt"] + 2.0 * mix[a]["ens"]),
                           rationale="multi-horizon marginal over dev duration prior")
            for a in CANDIDATES]


def select(estimates: list[ImpactEstimate], family: str | None = None,
           severity: float = 1.0, avoid: set[str] | None = None) -> ImpactEstimate:
    """Minimise predicted cost J = curt + ENS_WEIGHT * ens over the allowed set.

    ``no_op`` is forced when there is no incident evidence (family 'none' or
    severity below the action threshold) — the estimator ranks mitigations, it
    does not decide incident existence (that is the Evidence Gate's job).
    """
    avoid = avoid or set()
    pool = [e for e in estimates if e.action not in avoid] or list(estimates)
    if family == "none" or severity < 0.30:
        for e in pool:
            if e.action == "no_op":
                return e
    return min(pool, key=lambda e: e.expected_curt_kwh
               + ENS_WEIGHT * e.expected_ens_kwh)
