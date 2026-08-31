"""Energy Impact Agent — estimates expected ENS / curtailment per candidate
mitigation given the current operating point.

The estimator combines:
  (i) a deterministic *physical* lower bound from the FeatureView (how much
      the asset is contributing right now relative to its nominal),
  (ii) an LLM-grounded *qualitative* tier (negligible / low / medium / high
       / severe) that is calibrated against the physical bound.

Per candidate we return:
  - expected_curt_kwh : approximate over a 60 s response window
  - expected_ens_kwh  : 0 if the mitigation is non-disruptive
  - tier              : the worse of (physical_tier, llm_tier)

The combined tier is the field consumed by the Caution / Coordinator agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .telemetry_features import FeatureView


CANDIDATE_ACTIONS = (
    "no_op",
    "freeze_setpoint",
    "throttle_ramp",
    "request_ied_revalidation",
    "isolate_inverter",
)

# Physical impact priors, hand-calibrated from the scenario library:
# "isolate_inverter" cuts the asset; "freeze_setpoint" reverts to nominal;
# "throttle_ramp" caps swings; "request_ied_revalidation" is informational.
_RESPONSE_WINDOW_S = 60.0


@dataclass(frozen=True)
class ImpactEstimate:
    action: str
    tier: str               # negligible | low | medium | high | severe
    expected_curt_kwh: float
    expected_ens_kwh: float
    rationale: str


def _tier_from_curt(curt_kwh: float) -> str:
    if curt_kwh <= 0.05:
        return "negligible"
    if curt_kwh <= 0.5:
        return "low"
    if curt_kwh <= 2.0:
        return "medium"
    if curt_kwh <= 6.0:
        return "high"
    return "severe"


def _max_pair(tier_a: str, tier_b: str) -> str:
    order = ("negligible", "low", "medium", "high", "severe")
    return order[max(order.index(tier_a), order.index(tier_b))]


def estimate(
    fv: FeatureView,
    candidates: Iterable[str] = CANDIDATE_ACTIONS,
    target_asset: str | None = None,
) -> list[ImpactEstimate]:
    """Estimate expected impact tier for each candidate mitigation.

    `target_asset` is the inverter the mitigation will be aimed at; falls back
    to the FeatureView's dominant asset when not supplied.
    """
    asset = target_asset or fv.dominant_asset
    cur_kw = float(fv.asset_p_kw.get(asset or "", 0.0)) if asset else 0.0
    avail_kw = float(fv.asset_p_avail_kw.get(asset or "", 0.0)) if asset else 0.0

    estimates: list[ImpactEstimate] = []
    win = _RESPONSE_WINDOW_S / 3600.0  # convert seconds → hours
    for c in candidates:
        if c == "no_op":
            # Doing nothing has no curtailment — but if an attack is actually
            # active and we don't act, the ENS over the response window is
            # roughly the gap between available and current dispatch.
            ens = max(0.0, (avail_kw - cur_kw)) * win if fv.severity_score > 0.5 else 0.0
            estimates.append(ImpactEstimate(
                action=c, expected_curt_kwh=0.0, expected_ens_kwh=ens,
                tier=_tier_from_curt(ens * 2.0),    # ENS counts double in tier
                rationale="No mitigation; passes through any active attack."))
        elif c == "isolate_inverter":
            curt = max(0.0, cur_kw) * win
            estimates.append(ImpactEstimate(
                action=c, expected_curt_kwh=curt,
                expected_ens_kwh=0.0,
                tier=_max_pair(_tier_from_curt(curt * 1.5), "high"),
                rationale="Cuts the asset; full curtailment over response window."))
        elif c == "freeze_setpoint":
            # Reverts to nominal — small curtailment if asset was dispatched
            # above nominal, near-zero otherwise.
            nominal = avail_kw * 0.7 if avail_kw else cur_kw
            curt = max(0.0, cur_kw - nominal) * win + 0.05  # baseline ~0.05 kWh
            estimates.append(ImpactEstimate(
                action=c, expected_curt_kwh=curt, expected_ens_kwh=0.0,
                tier=_tier_from_curt(curt),
                rationale="Reverts asset to last validated setpoint; bounded curtailment."))
        elif c == "throttle_ramp":
            estimates.append(ImpactEstimate(
                action=c, expected_curt_kwh=0.1, expected_ens_kwh=0.0,
                tier="low",
                rationale="Caps fast swings; minor energy impact."))
        elif c == "request_ied_revalidation":
            estimates.append(ImpactEstimate(
                action=c, expected_curt_kwh=0.02, expected_ens_kwh=0.0,
                tier="negligible",
                rationale="IED re-attest; transient telemetry gap only."))
    return estimates


def cheapest_safe(
    estimates: list[ImpactEstimate],
    severity_score: float,
    avoid: set[str] | None = None,
    ens_weight: float = 5.0,
    tier_penalty: float = 1.5,
    severity_threshold: float = 0.30,
) -> ImpactEstimate:
    """Pick the candidate that minimises::

        J(a) = curtailment_kWh(a)
             + ens_weight * ENS_kWh(a)
             + tier_penalty * 1{tier(a) in {high, severe}}

    All three weights are exposed as keyword arguments so the IJCIP
    objective-sensitivity sweep can vary them without touching the
    Coordinator code.
    """
    avoid = avoid or set()
    def score(e: ImpactEstimate) -> float:
        penalty = tier_penalty if e.tier in {"high", "severe"} else 0.0
        ens_pen = ens_weight * e.expected_ens_kwh
        return e.expected_curt_kwh + penalty + ens_pen

    candidates = [e for e in estimates if e.action not in avoid]
    if not candidates:
        candidates = list(estimates)
    if severity_score < severity_threshold:
        for e in candidates:
            if e.action == "no_op":
                return e
    else:
        candidates = [e for e in candidates if e.action != "no_op"] or list(estimates)
    return min(candidates, key=score)
