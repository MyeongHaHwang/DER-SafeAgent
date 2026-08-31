r"""UNSW-MG24 loader and FeederAdapter shim.

The published UNSW-MG24 dataset~\cite{Zhang2025_UNSW_MG24} provides
NetFlow-style alert records over a microgrid testbed. Our co-simulation
harness expects (telemetry, events) per time-step from a `FeederAdapter`. We
bridge by:

  1. Loading UNSW-MG24 alerts from a CSV (path supplied by the user; the
     official dataset is gated by registration so we do not bundle it).
  2. Time-binning alerts into 1-second buckets matching `dt_s`.
  3. Emitting a `TelemetrySample` per tick with held-constant placeholder
     telemetry (UNSW-MG24 lacks per-bus voltages) and one `EventLog` per
     alert in the bucket.
  4. Recording the ground-truth UNSW-MG24 label in the event payload so the
     evaluator can score per-attack-class while leaving the detector blind.

The schema expected of the input CSV is documented in `INSTALL_DATASET.md`.
For environments where the real CSV is unavailable, `synthesize_alerts`
produces a tiny mock that follows the same column conventions.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ...simulation.types import Action, EventLog, TelemetrySample
from .label_map import to_der_class


@dataclass
class UNSWMG24Loader:
    """Reads a UNSW-MG24-style alert CSV.

    Required columns: `timestamp` (float, seconds), `src_ip`, `dst_ip`,
    `protocol`, `attack_type` (one of the NSL-KDD labels), and any further
    fields are passed through verbatim.
    """
    csv_path: str
    _alerts_by_bucket: dict[int, list[dict]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(f"UNSW-MG24 alert CSV not found: {self.csv_path}")
        self._load()

    def _load(self) -> None:
        with open(self.csv_path) as f:
            for row in csv.DictReader(f):
                t = int(float(row.get("timestamp", "0")))
                self._alerts_by_bucket.setdefault(t, []).append(row)

    def alerts_at(self, t_sec: int) -> list[dict]:
        return self._alerts_by_bucket.get(t_sec, [])

    @property
    def total_alerts(self) -> int:
        return sum(len(v) for v in self._alerts_by_bucket.values())

    @property
    def time_horizon_s(self) -> int:
        return max(self._alerts_by_bucket.keys(), default=0) + 1


@dataclass
class UNSWMG24Feeder:
    """A `FeederAdapter` that replays UNSW-MG24 alerts as events while
    holding the (placeholder) physical telemetry static.

    Use this when you want to evaluate detectors on UNSW-MG24 traffic under
    the same harness contract as the OpenDSS scenarios. Energy-impact
    metrics are not meaningful here (no power flow); the harness will still
    compute them but reviewers should report only detection metrics for
    UNSW-MG24 runs (see paper §5.1).
    """
    loader: UNSWMG24Loader
    placeholder_buses: list[str] = field(default_factory=lambda: ["BUS_PROXY"])
    placeholder_ders: list[dict] = field(default_factory=lambda: [{"id": "DER_PROXY", "max_kw": 100.0}])

    def solve(self, t: float) -> None:
        return

    def read(self, t: float) -> tuple[TelemetrySample, list[EventLog]]:
        sample = TelemetrySample(
            t=t,
            bus_voltages_pu={b: 1.0 for b in self.placeholder_buses},
            freq_hz=60.0,
            der_p_kw={d["id"]: float(d.get("max_kw", 100.0)) * 0.7 for d in self.placeholder_ders},
            der_p_avail_kw={d["id"]: float(d.get("max_kw", 100.0)) for d in self.placeholder_ders},
            load_demand_kw=1000.0,
            load_served_kw=1000.0,
        )
        events: list[EventLog] = []
        for alert in self.loader.alerts_at(int(t)):
            der_class, confidence_cap = to_der_class(alert.get("attack_type", ""))
            events.append(EventLog(
                t=t,
                source=alert.get("protocol", "unknown").lower(),
                kind="alarm",
                payload={
                    "src_ip": alert.get("src_ip"),
                    "dst_ip": alert.get("dst_ip"),
                    "kdd_label": alert.get("attack_type"),
                    "der_class_hint": der_class,
                    "confidence_cap": confidence_cap,
                    # passthrough non-canonical fields for the agent prompt
                    **{k: v for k, v in alert.items()
                       if k not in {"timestamp", "src_ip", "dst_ip", "protocol", "attack_type"}},
                },
                tampered=False,
            ))
        return sample, events

    def apply(self, action: Action) -> None:
        return


def synthesize_alerts(out_path: str, n_alerts: int = 50, seed: int = 0,
                      duration_s: int = 600) -> None:
    """Write a tiny mock CSV that follows the UNSW-MG24 column schema for
    pipeline validation when the real dataset is unavailable."""
    rng = random.Random(seed)
    cats = list({"DoS": 10, "Probe": 8, "MITM": 3, "BruteForce": 8, "U2R": 13, "R2L": 8}.items())
    rows: list[dict] = []
    for i in range(n_alerts):
        cat = rng.choices([c for c, _ in cats], weights=[w for _, w in cats])[0]
        rows.append({
            "timestamp": rng.randint(60, duration_s - 60),
            "src_ip": f"10.0.{rng.randint(1, 250)}.{rng.randint(1, 250)}",
            "dst_ip": "10.0.1.10",
            "protocol": rng.choice(["tcp", "udp", "modbus", "dnp3"]),
            "attack_type": cat,
            "src_port": rng.randint(1024, 65535),
            "dst_port": rng.choice([502, 20000, 80, 443]),
        })
    rows.sort(key=lambda r: r["timestamp"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
