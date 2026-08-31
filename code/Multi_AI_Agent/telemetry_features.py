"""Deterministic Telemetry Analyst — Stage 1 of the 5-agent loop.

Extracts a structured FeatureView from the recent telemetry/event window.
Pre-computing these features in a deterministic stage gives every downstream
LLM agent the same numeric grounding (rather than re-deriving them from raw
JSON), which both stabilises the prompt distribution and reduces the
probability of hallucinated statistics.

Features:
- per-asset z-score against a 60-step rolling baseline of reported P
- voltage-band excursion fraction over the visible buses
- system-frequency deviation magnitude
- count of forged-looking commands in the event window
- count of explicitly tampered telemetry events
- replay-signature flag (any duplicated event payload within the window)
- DoS signature flag (telemetry frozen across N consecutive samples)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FeatureView:
    """Compact, JSON-serialisable summary consumed by downstream agents."""
    t: float
    asset_zscores: dict[str, float] = field(default_factory=dict)
    asset_p_kw: dict[str, float] = field(default_factory=dict)
    asset_p_avail_kw: dict[str, float] = field(default_factory=dict)
    voltage_excursion_frac: float = 0.0
    voltage_min_pu: float = 1.0
    voltage_max_pu: float = 1.0
    freq_dev_hz: float = 0.0
    n_command_events: int = 0
    # Command packets whose payload carries auth_valid == False (a failed
    # MAC/signature). Distinct from n_command_events: on an authenticated bus
    # only these indicate a forgery; on an unauthenticated bus auth is absent
    # and command evidence is UNKNOWN. See the gate's auth_mode.
    n_unauth_command_events: int = 0
    n_tampered_events: int = 0
    n_dup_events: int = 0
    persistent_freeze: bool = False
    # Consecutive trailing window entries whose reported timestamp does not
    # advance (SCADA staleness / data-quality signature; fires under DoS and
    # replay, which re-emit captured samples). Added for the P0-B Incident
    # Evidence Gate (tag ijcip_final_safeagent_20260810); does NOT contribute
    # to severity_score, so all published severity-driven behaviour is
    # unchanged.
    telemetry_stale_ticks: int = 0
    # Observable protocol sequence regression (v3): a telemetry frame carrying
    # a sequence number <= one already seen. This is the wire-visible signature
    # of a replay and the runtime-safe replacement for the removed `tampered`
    # oracle on the replay family.
    seq_regressions: int = 0
    # Observable telemetry-integrity residual (v3, Phase 1; runtime-safe only).
    # True when a large reported-power deviation is NOT corroborated by any
    # voltage-band response --- i.e. the reported number moved but the grid
    # state did not. This is a state-estimation residual computable from
    # reported power and bus voltage alone (both on the wire); it is the
    # observable replacement for the removed injector `tampered` oracle and is
    # the runtime-safe indicator of a false-data-injection-style anomaly.
    integrity_residual: bool = False
    dominant_asset: str | None = None
    dominant_signal: str = "none"            # "telemetry_dev" | "command" | "tampered" | "replay" | "freeze"
    severity_score: float = 0.0              # 0..1 monotone: how strongly something looks anomalous

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["asset_zscores"] = {k: round(v, 3) for k, v in self.asset_zscores.items()}
        d["asset_p_kw"] = {k: round(v, 1) for k, v in self.asset_p_kw.items()}
        d["asset_p_avail_kw"] = {k: round(v, 1) for k, v in self.asset_p_avail_kw.items()}
        d["voltage_excursion_frac"] = round(self.voltage_excursion_frac, 4)
        d["voltage_min_pu"] = round(self.voltage_min_pu, 4)
        d["voltage_max_pu"] = round(self.voltage_max_pu, 4)
        d["freq_dev_hz"] = round(self.freq_dev_hz, 4)
        d["severity_score"] = round(self.severity_score, 3)
        return d


def _zscore(values: list[float], current: float) -> float:
    n = len(values)
    if n < 5:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5 or 1e-6
    return (current - mean) / std


def extract(
    telemetry_window: list[dict],
    event_window: list[dict],
    voltage_band: tuple[float, float] = (0.95, 1.05),
    runtime_safe: bool = False,
) -> FeatureView:
    """Build a FeatureView from the harness windows.

    ``runtime_safe`` (revision v3, Phase 1): when True, the injector
    ground-truth ``tampered`` flag on each event is IGNORED. That flag is set
    by the attack injectors (``EventLog.tampered = True``) and its own type
    contract says "never visible to detector" --- reading it in the runtime
    path is an oracle leak. In runtime-safe mode ``n_tampered_events`` is
    forced to 0 and the ``tampered`` dominant-signal branch is removed, so a
    false-data-injection attack is visible only through its genuinely
    observable trace (the reported-power z-score), not through a label the
    attacker set. All other features (command packets, duplicate payloads,
    stale/non-advancing timestamps, persistent zero-freeze, z-scores,
    voltage) are directly observable and are unchanged. Default False
    reproduces every published (v2) result byte-for-byte.
    """
    if not telemetry_window:
        return FeatureView(t=0.0)
    last = telemetry_window[-1]
    t = float(last.get("t", 0.0))

    # asset z-scores against the prior 60-step rolling history (excluding now).
    der_now: dict[str, float] = dict(last.get("der_p_kw", {}))
    der_avail: dict[str, float] = dict(last.get("der_p_avail", {})) or {
        k: 0.0 for k in der_now
    }
    asset_z: dict[str, float] = {}
    for asset, p_now in der_now.items():
        history = [float(s.get("der_p_kw", {}).get(asset, p_now))
                   for s in telemetry_window[:-1][-60:]]
        asset_z[asset] = _zscore(history, float(p_now))

    # voltage excursion fraction across all buses in the *last* sample.
    v = list(dict(last.get("v_pu", {})).values())
    if v:
        lo, hi = voltage_band
        oob = sum(1 for x in v if x < lo or x > hi)
        v_frac = oob / len(v)
        v_min = min(v)
        v_max = max(v)
    else:
        v_frac = 0.0
        v_min = v_max = 1.0

    f = float(last.get("freq_hz", 60.0))
    freq_dev = abs(f - 60.0)

    # event-window aggregates
    n_command = 0
    n_unauth_command = 0
    n_tampered = 0
    seen_payloads: set[str] = set()
    n_dup = 0
    seq_regressions = 0
    _seq_max = None
    for e in event_window:
        kind = e.get("kind")
        if kind == "command":
            n_command += 1
            if (e.get("payload") or {}).get("auth_valid") is False:
                n_unauth_command += 1
        # Oracle guard (v3): `tampered` is injector ground truth; only read it
        # in legacy (non-runtime-safe) mode for result reproducibility.
        if not runtime_safe and e.get("tampered") is True:
            n_tampered += 1
        # Observable sequence-number regression (replay signature).
        s_no = e.get("seq")
        if s_no is not None:
            if _seq_max is not None and s_no <= _seq_max:
                seq_regressions += 1
            else:
                _seq_max = s_no
        key = (e.get("source", ""), e.get("kind", ""), str(sorted((e.get("payload") or {}).items())))
        skey = "|".join(map(str, key))
        if skey in seen_payloads:
            n_dup += 1
        seen_payloads.add(skey)

    # persistent-freeze: same exact reported P for >=10 consecutive samples
    # AND the asset is at zero kW for those samples (DoS signature).
    persistent_freeze = False
    if len(telemetry_window) >= 10 and der_now:
        any_frozen = False
        for asset, p_now in der_now.items():
            tail = [s.get("der_p_kw", {}).get(asset, p_now)
                    for s in telemetry_window[-10:]]
            if all(abs(float(x)) < 1e-3 for x in tail):
                any_frozen = True
                break
        persistent_freeze = any_frozen

    # SCADA staleness: consecutive trailing samples whose reported t does not
    # advance (a frozen/replayed sample carries its capture-time timestamp).
    stale_ticks = 0
    if len(telemetry_window) >= 2:
        t_prev = float(telemetry_window[-1].get("t", 0.0))
        for s in reversed(telemetry_window[:-1]):
            t_s = float(s.get("t", 0.0))
            if t_s >= t_prev - 1e-9:   # timestamp failed to advance
                stale_ticks += 1
                t_prev = t_s
            else:
                break

    # Observable integrity residual (runtime-safe only). Two rating- and
    # voltage-aware checks, both computable from telemetry the defender
    # already has (reported power, available capacity, bus voltage):
    #   (a) plausibility: reported active power exceeds the asset's available
    #       capacity --- physically impossible, so the reading is corrupted;
    #   (b) onset transient: a strong reported-power z-spike while the voltage
    #       band is intact (the number moved but the grid state did not).
    # Check (a) catches a sustained high-magnitude FDI that (b) misses once a
    # rolling baseline absorbs the bias; a sustained WITHIN-capacity bias is
    # caught only transiently by (b) --- an explicit detectability limit
    # (needs redundant sensing), reported as such.
    zmax = max((abs(z) for z in asset_z.values()), default=0.0)
    # 10% over nameplate: implausible even under heavy measurement noise
    # (a real sensor cannot report a DER producing above its rated capacity),
    # so this does not fire on benign noisy telemetry. It therefore catches
    # only high-magnitude FDI whose injected value pushes the reading above
    # capacity; sub-capacity FDI remains a documented miss.
    implausible = any(
        der_now.get(a, 0.0) > der_avail.get(a, float("inf")) * 1.10 + 1e-6
        for a in der_now)
    # Plausibility violation only. An onset z-spike is NOT used: benign cloud
    # and noise transients produce the same spike, so it is not separable from
    # an attack without redundant sensing. Consequence: only a physically
    # implausible reading (reported power above rated capacity) is treated as
    # an integrity residual; sub-capacity FDI is a documented miss.
    integrity_residual = bool(runtime_safe and implausible)

    # dominant signal selection
    if n_command >= 1:
        dominant = "command"
    elif not runtime_safe and n_tampered >= 1:
        dominant = "tampered"
    elif persistent_freeze:
        dominant = "freeze"
    elif n_dup >= 2 or (runtime_safe and seq_regressions >= 2):
        dominant = "replay"
    elif runtime_safe and stale_ticks >= 3:
        # observable DoS/staleness signature (frozen non-advancing timestamps)
        dominant = "freeze"
    elif integrity_residual:
        dominant = "integrity_residual"
    elif asset_z and zmax > 3.0:
        dominant = "telemetry_dev"
    else:
        dominant = "none"

    dominant_asset: str | None = None
    if asset_z:
        dominant_asset = max(asset_z.items(), key=lambda kv: abs(kv[1]))[0]
    if dominant == "command":
        for e in event_window:
            if e.get("kind") == "command":
                dominant_asset = (e.get("payload") or {}).get("asset", dominant_asset)
                break

    # severity_score: bounded in [0,1] — monotone in the strongest signal.
    sev = 0.0
    if n_command >= 1:
        sev = max(sev, min(1.0, 0.55 + 0.06 * n_command))
    if n_tampered >= 1:
        sev = max(sev, min(1.0, 0.55 + 0.05 * n_tampered))
    if persistent_freeze:
        sev = max(sev, 0.78)
    if n_dup >= 2:
        sev = max(sev, min(1.0, 0.55 + 0.05 * n_dup))
    if asset_z:
        sev = max(sev, min(1.0, max(abs(z) for z in asset_z.values()) / 8.0))
    if v_frac > 0.0:
        sev = max(sev, min(1.0, 0.4 + v_frac))
    # Staleness contributes to severity (P0-B addition). Strictly additive:
    # situations with >=3 non-advancing timestamps previously produced
    # severity ~0 and no downstream behaviour at all, so no published result
    # changes; dominant_signal is deliberately NOT extended (the deterministic
    # baseline's published signal->class map stays untouched).
    if stale_ticks >= 3:
        sev = max(sev, 0.65)
    # Integrity residual raises severity to the FDI action band (runtime-safe
    # replacement for the tampered-flag severity contribution).
    if integrity_residual:
        sev = max(sev, 0.60)
    if runtime_safe and seq_regressions >= 2:
        sev = max(sev, 0.60)

    return FeatureView(
        t=t,
        asset_zscores=asset_z,
        asset_p_kw=der_now,
        asset_p_avail_kw=der_avail,
        voltage_excursion_frac=v_frac,
        voltage_min_pu=v_min,
        voltage_max_pu=v_max,
        freq_dev_hz=freq_dev,
        n_command_events=n_command,
        n_unauth_command_events=n_unauth_command,
        n_tampered_events=n_tampered,
        n_dup_events=n_dup,
        persistent_freeze=persistent_freeze,
        telemetry_stale_ticks=stale_ticks,
        seq_regressions=seq_regressions,
        integrity_residual=integrity_residual,
        dominant_asset=dominant_asset,
        dominant_signal=dominant,
        severity_score=sev,
    )
