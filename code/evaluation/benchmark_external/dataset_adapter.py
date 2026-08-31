"""Dataset adapters for the external-benchmark suite.

Each adapter exposes a uniform API:

    adapter = ExternalDatasetAdapter("unsw_mg24")
    if adapter.is_available():
        for sample in adapter.iter_examples():
            ...
    else:
        # the run script writes a "not run" row with adapter.unavailable_reason

Only UNSW-MG24 ships with a real adapter (with synthetic mock fallback).
The other adapters check for a CSV at a configurable path and otherwise
report "not available". This is deliberate: we do not bundle gated datasets
and we do not auto-download them.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .label_mapping import to_der_class

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ExternalSample:
    """Canonical example row presented to the detector / scorer."""
    sample_id: str
    dataset: str
    source_label: str
    der_class: str           # mapped via label_mapping
    confidence_cap: float    # source-label mapping precision
    payload: dict            # arbitrary fields kept for the detector prompt


@dataclass
class ExternalDatasetAdapter:
    name: str
    csv_path_override: str | None = None
    _path: Path | None = None
    _reason: str = ""

    def __post_init__(self) -> None:
        self._path = self._resolve()

    # ---------- path resolution ----------

    def _candidate_paths(self) -> list[Path]:
        if self.csv_path_override:
            return [Path(self.csv_path_override)]
        n = self.name.lower()
        if n == "unsw_mg24":
            return [
                REPO_ROOT / "code/datasets/unsw_mg24/raw/alerts.csv",
                REPO_ROOT / "code/datasets/unsw_mg24/raw/alerts_mock.csv",
            ]
        return [
            REPO_ROOT / f"code/datasets/{n}/raw/{n}.csv",
            REPO_ROOT / f"code/datasets/{n}/raw/labels.csv",
        ]

    def _resolve(self) -> Path | None:
        for p in self._candidate_paths():
            if p.exists():
                return p
        self._reason = (f"no CSV found for dataset {self.name!r}; "
                         "place a labelled CSV under "
                         f"code/datasets/{self.name}/raw/ and re-run")
        return None

    # ---------- public API ----------

    def is_available(self) -> bool:
        return self._path is not None

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    @property
    def path(self) -> str:
        return str(self._path) if self._path else ""

    def iter_examples(self) -> Iterator[ExternalSample]:
        if not self._path:
            return iter(())

        with self._path.open() as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                lbl = self._extract_label(row)
                der, cap = to_der_class(self.name, lbl)
                yield ExternalSample(
                    sample_id=f"{self.name}-{i:06d}",
                    dataset=self.name,
                    source_label=lbl,
                    der_class=der,
                    confidence_cap=cap,
                    payload=row,
                )

    @staticmethod
    def _extract_label(row: dict) -> str:
        for col in ("attack_type", "label", "Label", "Attack", "class", "Class"):
            if col in row and row[col] != "":
                return str(row[col])
        return "Normal"


# ---------- convenience: list all benchmarks ----------

ALL_BENCHMARKS: tuple[str, ...] = (
    "unsw_mg24", "swat", "wadi", "ton_iot", "epic",
)


def all_adapters(overrides: dict[str, str] | None = None) -> list[ExternalDatasetAdapter]:
    overrides = overrides or {}
    return [
        ExternalDatasetAdapter(name=n, csv_path_override=overrides.get(n))
        for n in ALL_BENCHMARKS
    ]
