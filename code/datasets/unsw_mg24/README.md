# UNSW-MG24 dataset adapter (production-grade)

Two backends are provided:

| Loader | Module | Status |
|---|---|---|
| Mock alert generator | `loader.py::synthesize_alerts` | always available; ships a 200-row CSV at `raw/alerts_mock.csv` |
| Production loader    | `production_loader.py::UnswProductionLoader` | resolves a CSV / Parquet on disk; falls back to the mock |

## Resolution order (production loader)

1. `--csv` / `--parquet` CLI override.
2. `UNSW_MG24_ROOT` environment variable. The variable can point at a
   labelled CSV / Parquet **file** or at a **directory** that contains
   one or more such files; the first match is used.
3. `code/datasets/unsw_mg24/raw/alerts.csv` (drop-in production CSV).
4. `code/datasets/unsw_mg24/raw/alerts_mock.csv` (shipped mock).

The loader records `result_source` on every sample
(`"production"` vs `"mock"`) so downstream artefacts can filter
mock-only or production-only.

## Required CSV columns

| Column | Type | Required | Notes |
|---|---|---|---|
| `timestamp`   | float | yes | seconds since the run start |
| `attack_type` / `label` / `class` | string | yes | label is auto-detected |

The loader's column inference handles the common UNSW-MG24 / NSL-KDD
column conventions. Pass `--label-col` / `--timestamp-col` to override.

## Label mapping

`label_mapping.py` exposes:
- `DER_CLASSES_EXTENDED = (none, fdi, replay, command_spoof, dos, firmware, unknown_or_unmapped)`
- `to_der_class(label)` --- production mapping with explicit
  abstention. Labels that have no high-confidence cyber-physical
  equivalent collapse to `unknown_or_unmapped` so the detector can
  abstain or route to HITL rather than force a packet-level label
  into a DER taxonomy entry it cannot act on safely.
- `to_der_class_strict(label)` --- the legacy 6-class collapse, used
  only for backward-compatible macro-F1.

## Validating a drop-in dataset

```bash
python -m code.datasets.unsw_mg24.validate_dataset \
    --path /path/to/your/alerts.csv \
    --max-samples 10000
```

The script prints a JSON summary with the inferred label column, the
label distribution, the DER-class distribution, and the
unknown-or-unmapped fraction.
