"""Production UNSW-MG24 loader.

Resolves a labelled CSV (or Parquet, or directory of shard files) on
disk and exposes a uniform iterator the external-benchmark scorer can
consume. The mock CSV shipped at ``raw/alerts_mock.csv`` is the
fallback when no production data is configured. ``result_source`` is
recorded on every emitted sample so that downstream artefacts can be
filtered to mock-only or production-only.

Resolution order:
    1. ``--csv`` CLI override on the run script
    2. ``UNSW_MG24_ROOT`` environment variable (CSV / Parquet / dir)
    3. ``code/datasets/unsw_mg24/raw/alerts.csv`` (production drop-in)
    4. ``code/datasets/unsw_mg24/raw/alerts_mock.csv`` (shipped mock)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .label_mapping import to_der_class

REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------- column inference ----------

LABEL_CANDIDATES = (
    "attack_type", "Attack", "attack", "label", "Label",
    "class", "Class", "category", "Category",
)
TIMESTAMP_CANDIDATES = ("timestamp", "ts", "time", "Time", "Stime")


def _pick_column(header: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in header:
            return c
    return None


# ---------- canonical sample ----------

@dataclass
class UnswSample:
    sample_id: str
    source_label: str
    der_class: str
    confidence_cap: float
    timestamp: float
    payload: dict
    result_source: str   # "production" | "mock"


# ---------- production loader ----------

@dataclass
class UnswProductionLoader:
    """Loads UNSW-MG24 records from a labelled CSV or Parquet file.

    The constructor never fabricates data. If no path is provided and
    no environment variable / fallback file exists, ``available()``
    returns ``False`` and ``unavailable_reason`` describes the
    resolution attempt.
    """
    csv_path: str | None = None
    parquet_path: str | None = None
    label_col: str | None = None
    timestamp_col: str | None = None
    _resolved_path: Path | None = None
    _result_source: str = "production"
    _reason: str = ""

    def __post_init__(self) -> None:
        self._resolved_path = self._resolve()

    def _resolve(self) -> Path | None:
        candidates: list[tuple[Path, str]] = []
        if self.parquet_path:
            candidates.append((Path(self.parquet_path), "production"))
        if self.csv_path:
            candidates.append((Path(self.csv_path), "production"))
        env = os.environ.get("UNSW_MG24_ROOT", "")
        if env:
            ep = Path(env)
            if ep.is_dir():
                # take the first labelled csv / parquet in the directory
                for p in sorted(ep.glob("*.csv")):
                    candidates.append((p, "production"))
                for p in sorted(ep.glob("*.parquet")):
                    candidates.append((p, "production"))
            elif ep.suffix in {".csv", ".parquet"}:
                candidates.append((ep, "production"))
        candidates.append(
            (REPO_ROOT / "code/datasets/unsw_mg24/raw/alerts.csv", "production"))
        candidates.append(
            (REPO_ROOT / "code/datasets/unsw_mg24/raw/alerts_mock.csv", "mock"))

        for p, src in candidates:
            if p.exists():
                self._result_source = src
                return p
        self._reason = (
            "no UNSW-MG24 file found on disk; tried "
            "UNSW_MG24_ROOT (unset/empty), "
            "code/datasets/unsw_mg24/raw/alerts.csv, "
            "code/datasets/unsw_mg24/raw/alerts_mock.csv"
        )
        return None

    # ----- public API -----

    def available(self) -> bool:
        return self._resolved_path is not None

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    @property
    def path(self) -> str:
        return str(self._resolved_path) if self._resolved_path else ""

    @property
    def result_source(self) -> str:
        return self._result_source

    def iter_samples(self, max_samples: int | None = None) -> Iterator[UnswSample]:
        if not self._resolved_path:
            return iter(())
        if self._resolved_path.suffix == ".parquet":
            yield from self._iter_parquet(max_samples)
        else:
            yield from self._iter_csv(max_samples)

    # ----- backends -----

    def _iter_csv(self, max_samples: int | None) -> Iterator[UnswSample]:
        with self._resolved_path.open() as fh:
            reader = csv.DictReader(fh)
            label_col = self.label_col or _pick_column(reader.fieldnames or [],
                                                          LABEL_CANDIDATES)
            ts_col = self.timestamp_col or _pick_column(reader.fieldnames or [],
                                                          TIMESTAMP_CANDIDATES)
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                lbl = str(row.get(label_col, "Normal")) if label_col else "Normal"
                ts = float(row.get(ts_col, "0")) if ts_col else 0.0
                der, cap = to_der_class(lbl)
                yield UnswSample(
                    sample_id=f"unsw_mg24-{i:06d}",
                    source_label=lbl,
                    der_class=der,
                    confidence_cap=cap,
                    timestamp=ts,
                    payload=dict(row),
                    result_source=self._result_source,
                )

    def _iter_parquet(self, max_samples: int | None) -> Iterator[UnswSample]:
        try:
            import pandas as pd
        except ImportError as exc:
            self._reason = f"pandas required for parquet: {exc}"
            return
        df = pd.read_parquet(self._resolved_path)
        cols = list(df.columns)
        label_col = self.label_col or _pick_column(cols, LABEL_CANDIDATES)
        ts_col = self.timestamp_col or _pick_column(cols, TIMESTAMP_CANDIDATES)
        for i, row in df.iterrows():
            if max_samples is not None and i >= max_samples:
                break
            lbl = str(row[label_col]) if label_col else "Normal"
            ts = float(row[ts_col]) if ts_col else 0.0
            der, cap = to_der_class(lbl)
            yield UnswSample(
                sample_id=f"unsw_mg24-{int(i):06d}",
                source_label=lbl,
                der_class=der,
                confidence_cap=cap,
                timestamp=ts,
                payload={c: row[c] for c in cols},
                result_source=self._result_source,
            )
