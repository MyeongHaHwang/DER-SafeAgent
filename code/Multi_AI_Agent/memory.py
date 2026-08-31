"""Incident memory store — retrieval-augmentation for the Hypothesis Agent.

Indexes past resolved decisions by an interpretable signature so that, when a
similar situation recurs, the Hypothesis Agent can be conditioned with the
prior verdict and its outcome. The store is intentionally simple: a JSONL
file with an in-memory cache, keyed by ``(scenario_class, dominant_signal,
dominant_asset_kind)``. No embedding model is needed; the keys are derived
from the deterministic FeatureView, which keeps the retrieval reproducible.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_lock = threading.Lock()


@dataclass
class IncidentRecord:
    decision_id: str
    signature: str
    attack_class: str
    affected_asset: str | None
    chosen_action: str
    outcome_tier: str            # "negligible" .. "severe"
    confidence: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class IncidentMemory:
    store_path: Path | None = None
    _records: list[IncidentRecord] = field(default_factory=list)
    _index: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store_path and self.store_path.exists():
            for line in self.store_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = IncidentRecord(**json.loads(line))
                    self._records.append(rec)
                    self._index.setdefault(rec.signature, []).append(len(self._records) - 1)
                except Exception:
                    continue

    @staticmethod
    def signature(dominant_signal: str, asset_kind: str) -> str:
        return f"{dominant_signal}|{asset_kind}"

    @staticmethod
    def asset_kind_from_id(asset_id: str | None) -> str:
        if not asset_id:
            return "unknown"
        s = asset_id.lower()
        if "bess" in s:
            return "bess"
        if "inv" in s or "pv" in s:
            return "pv"
        return "other"

    def lookup(self, dominant_signal: str, asset_kind: str, k: int = 3) -> list[IncidentRecord]:
        key = self.signature(dominant_signal, asset_kind)
        idxs = self._index.get(key, [])
        # most-recent-first (reverse), capped at k
        return [self._records[i] for i in reversed(idxs[-k:])]

    def append(self, rec: IncidentRecord) -> None:
        with _lock:
            self._records.append(rec)
            self._index.setdefault(rec.signature, []).append(len(self._records) - 1)
            if self.store_path is not None:
                self.store_path.parent.mkdir(parents=True, exist_ok=True)
                with self.store_path.open("a") as fh:
                    fh.write(json.dumps(rec.to_dict()) + "\n")
