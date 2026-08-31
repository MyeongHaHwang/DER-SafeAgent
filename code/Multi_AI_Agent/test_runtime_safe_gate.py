"""Phase-1 (v3) invariants: the runtime-safe path must never consume the
injector `tampered` ground-truth oracle, and each attack family must be
reachable through a genuinely observable signal (or documented as a limit).
"""
from __future__ import annotations

import yaml
from pathlib import Path

from .telemetry_features import extract
from .horizon_estimator import class_family
from ..simulation.attack_injectors import REGISTRY as ATT
from ..simulation.feeder import StubFeeder

BASE = Path(__file__).resolve().parents[1] / "simulation/scenarios/ijcip_revision"


def _window(cfg, upto, seed):
    f = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                   base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                   episode_seed=seed, emit_seq=True)
    injs = [ATT[a["type"]](**{k: v for k, v in a.items() if k != "type"})
            for a in cfg.get("attacks", [])]
    tw, ew = [], []
    for k in range(upto + 1):
        t = k * float(cfg["dt_s"])
        for j in injs:
            j.physical_mutate(t, f)
        f.solve(t)
        s, evs = f.read(t)
        for j in injs:
            s = j.mutate_telemetry(t, s)
            evs = j.mutate_events(t, evs)
        tw.append({"t": s.t, "freq_hz": s.freq_hz, "v_pu": s.bus_voltages_pu,
                   "der_p_kw": s.der_p_kw, "der_p_avail": s.der_p_avail_kw,
                   "load_demand_kw": s.load_demand_kw,
                   "load_served_kw": s.load_served_kw})
        ew += [{"t": e.t, "source": e.source, "kind": e.kind,
                "payload": e.payload, "tampered": e.tampered, "seq": e.seq}
               for e in evs]
    return tw[-60:], ew[-60:]


def _fv(name, upto=250, seed=7, runtime_safe=True):
    cfg = yaml.safe_load((BASE / name / "config.yaml").read_text())
    tw, ew = _window(cfg, upto, seed)
    return extract(tw, ew, runtime_safe=runtime_safe)


def test_runtime_safe_ignores_tampered_even_when_present():
    # FDI/replay events carry tampered=True; runtime-safe must not count them.
    fv = _fv("rv13_fdi_high_medium_nom_medpen_inv")
    assert fv.n_tampered_events == 0
    fv2 = _fv("rv13_replay_med_medium_nom_medpen_inv")
    assert fv2.n_tampered_events == 0


def test_flipping_tampered_flag_does_not_change_runtime_safe_view():
    """Adversary control over the `tampered` flag must not move any
    runtime-safe feature: set every event's tampered True vs False."""
    cfg = yaml.safe_load(
        (BASE / "rv13_replay_med_medium_nom_medpen_inv/config.yaml").read_text())
    tw, ew = _window(cfg, 250, 7)
    ew_true = [{**e, "tampered": True} for e in ew]
    ew_false = [{**e, "tampered": False} for e in ew]
    a = extract(tw, ew_true, runtime_safe=True)
    b = extract(tw, ew_false, runtime_safe=True)
    assert a.dominant_signal == b.dominant_signal
    assert a.severity_score == b.severity_score
    assert a.seq_regressions == b.seq_regressions


def test_command_spoof_observable_via_command_packet():
    fv = _fv("rv13_spoof_med_medium_nom_medpen_inv")
    assert fv.n_command_events >= 1 and fv.dominant_signal == "command"


def test_replay_observable_via_seq_regression_at_onset():
    # Within-window regression is visible at the replay onset boundary.
    fv = _fv("rv13_replay_med_medium_nom_medpen_inv", upto=190)
    assert fv.seq_regressions >= 2 and class_family(fv) == "stale"


def test_replay_observable_via_stateful_gate_hwm():
    # Sustained replay: the gate's persistent seq high-water mark catches it
    # even after the rolling window fills with (increasing) replayed seqs.
    from .evidence_gate import GatedDetector, GateParams
    from ..simulation.types import TelemetrySample

    class _Sink:
        name = "sink"
        _last_decision = None
        def step(self, t, tel, ev):
            from ..simulation.types import Detection
            return Detection(asset=None, attack_class="none", confidence=0.0), []

    cfg = yaml.safe_load(
        (BASE / "rv13_replay_med_medium_nom_medpen_inv/config.yaml").read_text())
    det = GatedDetector(inner=_Sink(), params=GateParams(p_hard=5, p_soft=10_000,
                        z_soft=3.0), runtime_safe=True)
    f = StubFeeder(monitored_buses=cfg["monitored_buses"], ders=cfg["ders"],
                   base_load_kw=float(cfg.get("base_load_kw", 1000.0)),
                   episode_seed=7, emit_seq=True)
    injs = [ATT[a["type"]](**{k: v for k, v in a.items() if k != "type"})
            for a in cfg.get("attacks", [])]
    opened_during_attack = False
    for k in range(301):
        t = float(k)
        for j in injs:
            j.physical_mutate(t, f)
        f.solve(t)
        s, evs = f.read(t)
        for j in injs:
            s = j.mutate_telemetry(t, s)
            evs = j.mutate_events(t, evs)
        det.step(t, s, evs)
        if 180 <= t <= 300 and det._gate.open_:
            opened_during_attack = True
    assert opened_during_attack, "gate never opened on sustained replay"


def test_dos_observable_via_staleness():
    fv = _fv("rv13_dos_med_medium_nom_medpen_inv")
    assert fv.telemetry_stale_ticks >= 3


def test_high_magnitude_fdi_observable_via_plausibility():
    fv = _fv("rv13_fdi_high_medium_nom_medpen_inv")
    assert fv.integrity_residual and class_family(fv) == "fdi_like"


def test_within_capacity_fdi_is_a_documented_limit_not_a_false_positive():
    # med/low FDI stays within capacity and is NOT detected when sustained;
    # this is a stated limitation, but it must not be a benign false positive.
    fv_med = _fv("rv13_fdi_med_medium_nom_medpen_inv")
    fv_benign = _fv("rv13_benign_none_none_nom_medpen_none")
    assert not fv_med.integrity_residual  # honest miss, not fabricated catch
    assert fv_benign.dominant_signal == "none"  # no benign false evidence


def test_benign_has_no_observable_attack_evidence():
    fv = _fv("rv13_benign_none_none_nom_medpen_none")
    assert fv.n_command_events == 0 and fv.seq_regressions == 0
    assert not fv.integrity_residual and fv.telemetry_stale_ticks < 3
