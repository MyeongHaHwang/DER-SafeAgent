"""Incident Evidence Gate (P0-B, tag ijcip_final_safeagent_20260810).

The runtime-assurance shield constrains WHICH action may execute, but the
prior evaluation showed it has no mechanism for deciding that NO incident
exists: on benign inputs the model asserts an attack often enough to trigger
unnecessary mitigation. This gate sits BEFORE the LLM-assisted mitigation
path and answers incident existence from deterministic evidence only.

Two evidence tiers, each with a temporal-persistence requirement:

- HARD (protocol-level; essentially attack-specific): tampered telemetry
  flags, forged command events, duplicated/stale telemetry (replay/DoS
  signature), persistent zero-freeze. Sustained for >= p_hard consecutive
  evidence ticks.
- SOFT (physical-only; shared with benign transients such as cloud shadows
  and measurement noise): rolling z-score excursion or voltage-band
  excursion. Sustained for >= p_soft consecutive ticks at >= z_soft.

Gate CLOSED  => no mitigation may execute and the LLM is not consulted
               (monitor only).
Gate OPEN    => the pipeline proceeds (hypothesis -> runtime-assurance shield).

The gate never reads free-form LLM rationale, model confidence, or any other
attacker-influenceable model output. Thresholds are calibrated on a
development split and frozen before held-out testing
(``code/configs/ijcip_final_safeagent_20260810/evidence_gate_frozen.json``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GateParams:
    p_hard: int = 2      # consecutive hard-evidence ticks required
    p_soft: int = 8      # consecutive soft-evidence ticks required
    z_soft: float = 3.5  # |z| threshold for soft evidence

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        d = {"p_hard": self.p_hard, "p_soft": self.p_soft,
             "z_soft": self.z_soft, **(extra or {})}
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "GateParams":
        d = json.loads(Path(path).read_text())
        return cls(p_hard=int(d["p_hard"]), p_soft=int(d["p_soft"]),
                   z_soft=float(d["z_soft"]))


@dataclass
class EvidenceGate:
    """Stateful per-run gate; feed one FeatureView per triggered tick."""
    params: GateParams = field(default_factory=GateParams)
    _hard_run: int = 0
    _soft_run: int = 0
    open_: bool = False
    last_reason: str = "no evidence"

    def update(self, fv) -> bool:
        g = (lambda k, d=0: (fv.get(k, d) if isinstance(fv, dict)
                             else getattr(fv, k, d)))
        # Hard evidence = genuinely observable protocol/state signatures.
        # `n_tampered_events` (injector ground truth) is only present in legacy
        # FeatureViews; the runtime-safe FeatureView sets it to 0 and instead
        # exposes `integrity_residual` (an observable state-estimation residual)
        # for the FDI family. Both are accepted so one gate serves both modes.
        hard = bool(g("n_tampered_events") or g("integrity_residual")
                    or g("n_command_events")
                    or g("persistent_freeze") or g("n_dup_events") >= 2
                    or g("telemetry_stale_ticks") >= 3
                    or g("seq_regressions") >= 2)
        z = g("asset_zscores", {}) or {}
        zmax = max((abs(float(v)) for v in z.values()), default=0.0)
        soft = bool(zmax >= self.params.z_soft
                    or g("voltage_excursion_frac", 0.0) > 0.0)

        self._hard_run = self._hard_run + 1 if hard else 0
        self._soft_run = self._soft_run + 1 if soft else 0

        if self._hard_run >= self.params.p_hard:
            self.open_, self.last_reason = True, (
                f"hard evidence sustained {self._hard_run} ticks")
        elif self._soft_run >= self.params.p_soft:
            self.open_, self.last_reason = True, (
                f"soft evidence sustained {self._soft_run} ticks "
                f"(z>={self.params.z_soft})")
        else:
            self.open_, self.last_reason = False, (
                f"insufficient evidence (hard {self._hard_run}/{self.params.p_hard}, "
                f"soft {self._soft_run}/{self.params.p_soft})")
        return self.open_


@dataclass
class GatedDetector:
    """Wrap any Detector: suppress mitigation while the gate is closed.

    The wrapper recomputes the deterministic FeatureView from its own rolling
    windows (the gate must not depend on the inner detector's internals) and
    passes actions through unchanged once the gate is open. While closed, the
    inner detector is NOT stepped, so no LLM call can occur — the gate bounds
    both unnecessary physical action and unnecessary model consultation.
    """
    inner: object
    params: GateParams
    name: str = ""
    # v3 Phase 1: when True the gate's FeatureView ignores the injector
    # `tampered` oracle and uses the observable integrity residual instead.
    runtime_safe: bool = False
    _gate: EvidenceGate = field(default=None, repr=False)
    _telemetry: list = field(default_factory=list, repr=False)
    _events: list = field(default_factory=list, repr=False)
    _seq_hwm: int = field(default=-1, repr=False)   # persistent seq high-water mark
    n_suppressed_steps: int = 0
    n_open_steps: int = 0

    def __post_init__(self) -> None:
        self.name = self.name or f"gated_{getattr(self.inner, 'name', 'det')}"
        self._gate = EvidenceGate(self.params)

    @property
    def _last_decision(self):
        return getattr(self.inner, "_last_decision", None)

    def step(self, t, telemetry, events):
        from ..simulation.types import Detection
        from .telemetry_features import extract

        self._telemetry.append({
            "t": telemetry.t, "freq_hz": telemetry.freq_hz,
            "v_pu": telemetry.bus_voltages_pu, "der_p_kw": telemetry.der_p_kw,
            "der_p_avail": telemetry.der_p_avail_kw,
            "load_demand_kw": telemetry.load_demand_kw,
            "load_served_kw": telemetry.load_served_kw})
        self._telemetry = self._telemetry[-60:]
        self._events += [{"t": e.t, "source": e.source, "kind": e.kind,
                          "payload": e.payload, "tampered": e.tampered, "seq": getattr(e, "seq", None)}
                         for e in events]
        self._events = self._events[-60:]

        fv = extract(telemetry_window=self._telemetry, event_window=self._events,
                     runtime_safe=self.runtime_safe)
        # Persistent sequence high-water mark (runtime-safe replay signature):
        # a within-window regression fades once the window slides fully into
        # replayed frames, so track the max seq ever seen and count frames in
        # THIS batch whose seq falls below it. Observable and stateful.
        if self.runtime_safe:
            batch_regressions = 0
            for e in events:
                s_no = getattr(e, "seq", None)
                if s_no is None:
                    continue
                if s_no <= self._seq_hwm:
                    batch_regressions += 1
                else:
                    self._seq_hwm = s_no
            if batch_regressions:
                fv.seq_regressions = max(fv.seq_regressions, batch_regressions + 1)
        if not self._gate.update(fv):
            self.n_suppressed_steps += 1
            return Detection(asset=None, attack_class="none", confidence=0.0,
                             rationale=f"evidence_gate closed: "
                                       f"{self._gate.last_reason}"), []
        self.n_open_steps += 1
        return self.inner.step(t, telemetry, events)
