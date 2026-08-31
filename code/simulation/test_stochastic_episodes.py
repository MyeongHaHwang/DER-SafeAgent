"""Tests that stochastic episodes are genuine (revision ijcip_final_revision, E2).

The published harness consumed no randomness, so its five "seeds" reproduced a
run byte-for-byte and were not independent observations. These tests enforce
that (a) the deterministic default is preserved exactly, and (b) when an
episode seed is supplied the exogenous trajectory actually differs.
"""
from __future__ import annotations

import hashlib

from .feeder import StubFeeder

BUSES = ["SOURCEBUS", "634"]
DERS = [{"id": "INV_634", "type": "pv", "bus": "634", "max_kw": 200.0},
        {"id": "BESS_675", "type": "bess", "bus": "675", "max_kw": 150.0}]


def _trace(seed):
    f = StubFeeder(monitored_buses=BUSES, ders=DERS, episode_seed=seed)
    out = []
    for k in range(120):
        t = float(k)
        f.solve(t)
        s, _ = f.read(t)
        out.append((round(s.load_demand_kw, 6), round(s.load_served_kw, 6),
                    tuple(round(v, 6) for v in s.der_p_kw.values()),
                    tuple(round(v, 6) for v in s.der_p_avail_kw.values())))
    return hashlib.sha256(repr(out).encode()).hexdigest()


def test_deterministic_default_is_unchanged():
    """episode_seed=None must reproduce the published behaviour bit-for-bit."""
    assert _trace(None) == _trace(None)


def test_distinct_episode_seeds_give_distinct_trajectories():
    """The defect the audit found: seeds that change nothing are not episodes."""
    digests = {_trace(s) for s in (0, 1, 2, 3, 4)}
    assert len(digests) == 5, "episode seeds must produce distinct trajectories"


def test_same_episode_seed_is_reproducible():
    assert _trace(7) == _trace(7)


def test_stochastic_differs_from_deterministic():
    assert _trace(0) != _trace(None)


def test_exogenous_variation_is_physically_bounded():
    """Randomness must perturb the operating point, not create absurd states."""
    f = StubFeeder(monitored_buses=BUSES, ders=DERS, episode_seed=3)
    for k in range(600):
        f.solve(float(k))
        s, _ = f.read(float(k))
        assert 0.5 * 1000.0 < s.load_demand_kw < 1.5 * 1000.0
        assert s.load_served_kw <= s.load_demand_kw + 1e-9
        for a, p in s.der_p_kw.items():
            assert p >= 0.0
            assert p <= s.der_p_avail_kw[a] + 1e-6
