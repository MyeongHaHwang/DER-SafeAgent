"""Feeder adapter — wraps OpenDSS so the harness has a clean interface.

Two implementations:
- `OpenDSSFeeder` — real OpenDSSDirect.py backend.
- `StubFeeder` — synthetic, used for tests when OpenDSS isn't installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .types import Action, EventLog, TelemetrySample


class FeederAdapter(Protocol):
    def solve(self, t: float) -> None: ...
    def read(self, t: float) -> tuple[TelemetrySample, list[EventLog]]: ...
    def apply(self, action: Action) -> None: ...


@dataclass
class OpenDSSFeeder:
    dss_path: str
    monitored_buses: list[str]
    ders: list[dict]   # [{id, type, max_kw, ...}]
    # Optional DSS commands executed once after Redirect (e.g. load scaling
    # ``Edit Load.l634 kW=...`` or DER rating changes for configuration
    # sweeps). Empty by default so published behaviour is unchanged.
    init_commands: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            import opendssdirect as dss  # type: ignore
        except ImportError as e:
            raise RuntimeError("OpenDSSDirect.py is required for OpenDSSFeeder") from e
        self._dss = dss
        self.n_solves = 0
        self.n_nonconverged = 0
        dss.Basic.ClearAll()
        dss.Command(f"Redirect {self.dss_path}")
        for cmd in self.init_commands:
            dss.Command(cmd)
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError(
                f"OpenDSS initial solve did not converge for {self.dss_path} "
                f"with init_commands={self.init_commands}")

    def solve(self, t: float) -> None:
        # advance time-series solution one step (yearly mode could be used; here we
        # use snapshot for simplicity in the pilot — switch to "Daily"/time-series in
        # production runs).
        self._dss.Solution.Solve()
        self.n_solves += 1
        if not self._dss.Solution.Converged():
            self.n_nonconverged += 1

    def read(self, t: float) -> tuple[TelemetrySample, list[EventLog]]:
        dss = self._dss
        bus_v: dict[str, float] = {}
        for b in self.monitored_buses:
            dss.Circuit.SetActiveBus(b)
            mags = dss.Bus.puVmagAngle()[::2]
            bus_v[b] = float(sum(mags) / max(len(mags), 1))

        der_p: dict[str, float] = {}
        der_avail: dict[str, float] = {}
        for d in self.ders:
            dss.Generators.Name(d["id"])
            powers = dss.CktElement.Powers()
            # OpenDSS returns [P1,Q1,P2,Q2,P3,Q3,...] for all phases & terminals;
            # generation convention is negative, so flip sign and sum P over phase entries.
            phase_p = sum(powers[i] for i in range(0, min(6, len(powers)), 2))
            der_p[d["id"]] = -float(phase_p)
            der_avail[d["id"]] = float(d.get("max_kw", der_p[d["id"]]))

        total_load = 0.0
        if dss.Loads.First() > 0:
            while True:
                total_load += float(dss.Loads.kW())
                if dss.Loads.Next() == 0:
                    break
        sample = TelemetrySample(
            t=t,
            bus_voltages_pu=bus_v,
            freq_hz=60.0,        # OpenDSS is steady-state; freq from external freq model
            der_p_kw=der_p,
            der_p_avail_kw=der_avail,
            load_demand_kw=total_load,
            load_served_kw=total_load,
        )
        events = self._emit_events(t, sample)
        return sample, events

    def _emit_events(self, t: float, sample: TelemetrySample) -> list[EventLog]:
        events: list[EventLog] = []
        for asset, p in sample.der_p_kw.items():
            events.append(EventLog(t=t, source="iec61850", kind="telemetry",
                                   payload={"asset": asset, "p_kw": p}))
        return events

    def apply(self, action: Action) -> None:
        dss = self._dss
        if action.name == "isolate_inverter":
            # ``Open Generator`` in OpenDSS only opens the terminal connection
            # but the next snapshot solve still credits the unit's generation.
            # Disabling the element is the correct way to remove its
            # contribution from the solution.
            dss.Command(f"Edit Generator.{action.target} enabled=no")
        elif action.name == "throttle_ramp":
            dss.Command(f"Edit InvControl.{action.target} RiseFallLimit=1.667")
        elif action.name == "freeze_setpoint":
            for d in self.ders:
                if d["id"] == action.target:
                    nominal = float(d.get("max_kw", 0.0)) * 0.7
                    dss.Command(f"Edit Generator.{action.target} kW={nominal}")
                    break
        elif action.name in ("request_ied_revalidation", "no_op"):
            pass
        else:
            raise ValueError(f"unknown action: {action.name}")

    def set_generator_kw(self, asset_id: str, kw: float) -> None:
        """Used by attack injectors to perturb the physical setpoint of a DER."""
        self._dss.Command(f"Edit Generator.{asset_id} kW={kw}")


@dataclass
class StubFeeder:
    """Deterministic synthetic feeder for tests / smoke-runs without OpenDSS.

    Tracks per-asset dispatch state so attack injectors (via
    ``set_generator_kw``) and detector actions (via ``apply``) are reflected
    in subsequent telemetry reads. Without this state, the cost-curves cannot
    distinguish methods because actions would have no physical effect on the
    time series.
    """
    monitored_buses: list[str]
    ders: list[dict]
    base_load_kw: float = 1000.0
    # --- genuine exogenous stochasticity (revision ijcip_final_revision, E2) ---
    # The published harness consumed no randomness: load and PV were constants
    # and there was no measurement noise, so every "seed" reproduced a run
    # byte-for-byte and repeated seeds were not independent observations.
    # ``episode_seed`` draws an actual exogenous trajectory: a diurnal load
    # shape with a random daily level, a PV availability profile with random
    # cloud events, and per-sample measurement noise. ``None`` reproduces the
    # published deterministic behaviour exactly, so legacy results remain
    # reproducible.
    episode_seed: int | None = None
    load_sigma: float = 0.08          # s.d. of the daily load level
    pv_cloud_rate: float = 0.004      # per-tick probability of a cloud event
    meas_noise_kw: float = 0.8        # s.d. of per-sample measurement noise
    # v3: stamp an observable monotonic sequence number on emitted telemetry
    # events so a replay's stale seq is detectable. Off by default (legacy
    # events carry seq=None and are byte-identical to v2).
    emit_seq: bool = False
    _seq: int = 0
    _state: dict = field(default_factory=dict)
    _enabled: dict = field(default_factory=dict)
    _frozen_setpoint: dict = field(default_factory=dict)
    _rng: object = field(default=None, repr=False)
    _load_scale: float = 1.0
    _pv_scale: float = 1.0
    _cloud_ticks: int = 0

    def _draw_exogenous(self, t: float) -> None:
        """Advance the exogenous state one tick. No-op when deterministic."""
        if self._rng is None:
            return
        # Diurnal load shape around the drawn daily level (600 s window).
        import math
        self._load_scale = self._daily_level * (1.0 + 0.05 * math.sin(2 * math.pi * t / 600.0))
        # PV: Poisson-ish cloud events that depress availability for a spell.
        if self._cloud_ticks > 0:
            self._cloud_ticks -= 1
            self._pv_scale = self._cloud_depth
        elif self._rng.random() < self.pv_cloud_rate:
            self._cloud_ticks = int(self._rng.integers(10, 60))
            self._cloud_depth = float(self._rng.uniform(0.45, 0.85))
            self._pv_scale = self._cloud_depth
        else:
            self._pv_scale = 1.0

    def __post_init__(self) -> None:
        if self.episode_seed is not None:
            import numpy as _np
            self._rng = _np.random.default_rng(self.episode_seed)
            self._daily_level = float(self._rng.normal(1.0, self.load_sigma))
            self._cloud_depth = 1.0
        self._state = {d["id"]: float(d.get("max_kw", 100.0)) * 0.7 for d in self.ders}
        self._enabled = {d["id"]: True for d in self.ders}
        self._frozen_setpoint = {}

    def solve(self, t: float) -> None:
        self._draw_exogenous(t)
        # If a frozen setpoint is in effect, restore the asset to its frozen kW.
        for asset, kw in list(self._frozen_setpoint.items()):
            if self._enabled.get(asset, True):
                self._state[asset] = kw

    def read(self, t: float) -> tuple[TelemetrySample, list[EventLog]]:
        max_total = sum(float(d.get("max_kw", 100.0)) for d in self.ders) or 1.0
        cur_total = sum(self._state[d["id"]] if self._enabled[d["id"]] else 0.0
                        for d in self.ders)
        nominal_total = max_total * 0.7
        deficit_frac = max(0.0, (nominal_total - cur_total) / max(nominal_total, 1.0))
        # Linear soft voltage proxy: drop scales with injection deficit.
        v_drop = min(0.06, 0.04 * deficit_frac)
        bus_v = {b: 1.0 - v_drop for b in self.monitored_buses}

        der_p = {d["id"]: (self._state[d["id"]] if self._enabled[d["id"]] else 0.0)
                 for d in self.ders}
        der_avail = {d["id"]: float(d.get("max_kw", 100.0)) for d in self.ders}
        if self._rng is not None:
            # PV availability follows the cloud process; reported power carries
            # measurement noise. Both are exogenous to the detector's actions.
            der_avail = {k: v * self._pv_scale for k, v in der_avail.items()}
            der_p = {k: max(0.0, min(v, der_avail[k])
                            + float(self._rng.normal(0.0, self.meas_noise_kw)))
                     for k, v in der_p.items()}
        # ENS only accumulates if total injection sustainedly drops below
        # ~60 % of nominal (deficit_frac > 0.4): proxy for backstop import.
        demand = self.base_load_kw * self._load_scale
        served = demand * (1.0 - 0.5 * deficit_frac if deficit_frac > 0.4 else 1.0)
        sample = TelemetrySample(
            t=t, bus_voltages_pu=bus_v, freq_hz=60.0,
            der_p_kw=der_p, der_p_avail_kw=der_avail,
            load_demand_kw=demand, load_served_kw=served,
        )
        events = []
        for k, v in der_p.items():
            seq = None
            if self.emit_seq:
                self._seq += 1
                seq = self._seq
            events.append(EventLog(t=t, source="iec61850", kind="telemetry",
                                   payload={"asset": k, "p_kw": v}, seq=seq))
        return sample, events

    def set_generator_kw(self, asset_id: str, kw: float) -> None:
        if asset_id in self._state:
            self._state[asset_id] = float(kw)

    def apply(self, action: Action) -> None:
        target = action.target
        if not target or target not in self._state:
            return
        if action.name == "isolate_inverter":
            self._enabled[target] = False
        elif action.name == "freeze_setpoint":
            for d in self.ders:
                if d["id"] == target:
                    nom = float(d.get("max_kw", 0.0)) * 0.7
                    self._frozen_setpoint[target] = nom
                    self._state[target] = nom
                    break
        elif action.name == "throttle_ramp":
            for d in self.ders:
                if d["id"] == target:
                    nom = float(d.get("max_kw", 0.0)) * 0.7
                    self._state[target] = 0.5 * (self._state[target] + nom)
                    break
        elif action.name in ("request_ied_revalidation", "no_op"):
            return
