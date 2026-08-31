"""Production ToN-IoT loader (network-flow split).

Resolves a labelled CSV or Parquet from disk and exposes a uniform
iterator over canonical samples. ``result_source`` records whether the
underlying file is a production dump or the small dev mock.

Resolution order:
    1. ``--csv`` / ``--parquet`` CLI override on the run script
    2. ``TON_IOT_ROOT`` environment variable (CSV / Parquet / directory)
    3. ``code/datasets/ton-iot/NF-ToN-IoT.parquet`` (NetFlow v2 bundle)
    4. ``code/datasets/ton-iot/test-flow.csv`` (CICFlow test split)

Label mapping reuses the UNSW-MG24 high-confidence cyber-physical
taxonomy so the abstention semantics match across benchmarks.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..unsw_mg24.label_mapping import to_der_class

REPO_ROOT = Path(__file__).resolve().parents[3]

LABEL_CANDIDATES = (
    "Attack", "attack_name", "attack", "label", "Label",
    "class", "Class", "category", "Category",
)
TIMESTAMP_CANDIDATES = ("timestamp", "ts", "time", "Time", "Stime",
                        "FLOW_DURATION_MILLISECONDS",
                        "flow_duration_milliseconds")


def _pick_column(header: list[str], candidates) -> str | None:
    for c in candidates:
        if c in header:
            return c
    return None


@dataclass
class TonIotSample:
    sample_id: str
    source_label: str
    der_class: str
    confidence_cap: float
    timestamp: float
    payload: dict
    result_source: str


@dataclass
class TonIotProductionLoader:
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
        cands: list[tuple[Path, str]] = []
        if self.parquet_path:
            cands.append((Path(self.parquet_path), "production"))
        if self.csv_path:
            cands.append((Path(self.csv_path), "production"))
        env = os.environ.get("TON_IOT_ROOT", "")
        if env:
            ep = Path(env)
            if ep.is_dir():
                for p in sorted(ep.glob("*.parquet")):
                    cands.append((p, "production"))
                for p in sorted(ep.glob("*.csv")):
                    cands.append((p, "production"))
            elif ep.suffix in {".csv", ".parquet"}:
                cands.append((ep, "production"))
        cands.append((REPO_ROOT / "code/datasets/ton-iot/NF-ToN-IoT.parquet",
                       "production"))
        cands.append((REPO_ROOT / "code/datasets/ton-iot/test-flow.csv",
                       "production"))
        for p, src in cands:
            if p.exists():
                self._result_source = src
                return p
        self._reason = (
            "no ToN-IoT file found on disk; tried TON_IOT_ROOT, "
            "code/datasets/ton-iot/NF-ToN-IoT.parquet, "
            "code/datasets/ton-iot/test-flow.csv"
        )
        return None

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

    def iter_samples(self, max_samples: int | None = None,
                       stratify_seed: int | None = 0) -> Iterator[TonIotSample]:
        if not self._resolved_path:
            return iter(())
        if self._resolved_path.suffix == ".parquet":
            yield from self._iter_parquet(max_samples, stratify_seed)
        else:
            yield from self._iter_csv(max_samples)

    def _iter_csv(self, max_samples: int | None) -> Iterator[TonIotSample]:
        with self._resolved_path.open() as fh:
            reader = csv.DictReader(fh)
            label_col = self.label_col or _pick_column(
                reader.fieldnames or [], LABEL_CANDIDATES)
            ts_col = self.timestamp_col or _pick_column(
                reader.fieldnames or [], TIMESTAMP_CANDIDATES)
            for i, row in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                lbl = str(row.get(label_col, "benign")) if label_col else "benign"
                ts = 0.0
                if ts_col:
                    try:
                        ts = float(row.get(ts_col, "0"))
                    except (TypeError, ValueError):
                        ts = 0.0
                der, cap = to_der_class(lbl)
                yield TonIotSample(
                    sample_id=f"ton_iot-{i:08d}",
                    source_label=lbl,
                    der_class=der,
                    confidence_cap=cap,
                    timestamp=ts,
                    payload=dict(row),
                    result_source=self._result_source,
                )

    def _iter_parquet(self, max_samples: int | None,
                        stratify_seed: int | None) -> Iterator[TonIotSample]:
        try:
            import pandas as pd
        except ImportError as exc:
            self._reason = f"pandas required for parquet: {exc}"
            return
        df = pd.read_parquet(self._resolved_path)
        cols = list(df.columns)
        label_col = self.label_col or _pick_column(cols, LABEL_CANDIDATES)
        ts_col = self.timestamp_col or _pick_column(cols, TIMESTAMP_CANDIDATES)
        if max_samples is not None and len(df) > max_samples and label_col:
            # Stratified sample so every attack class is represented
            seed = stratify_seed or 0
            classes = df[label_col].unique()
            per_class = max(1, max_samples // max(len(classes), 1))
            idx_pieces = []
            for cls in classes:
                grp_idx = df.index[df[label_col] == cls]
                k = min(len(grp_idx), per_class)
                if k <= 0:
                    continue
                idx_pieces.append(
                    pd.Series(grp_idx).sample(n=k, random_state=seed).tolist())
            keep_idx = [i for lst in idx_pieces for i in lst]
            df = df.loc[keep_idx].head(max_samples).reset_index(drop=True)
        elif max_samples is not None:
            df = df.head(max_samples).reset_index(drop=True)
        for i, row in df.iterrows():
            lbl = str(row[label_col]) if label_col else "benign"
            ts = 0.0
            if ts_col:
                try:
                    ts = float(row[ts_col])
                except (TypeError, ValueError):
                    ts = 0.0
            der, cap = to_der_class(lbl)
            yield TonIotSample(
                sample_id=f"ton_iot-{int(i):08d}",
                source_label=lbl,
                der_class=der,
                confidence_cap=cap,
                timestamp=ts,
                payload={c: row[c] for c in cols},
                result_source=self._result_source,
            )
