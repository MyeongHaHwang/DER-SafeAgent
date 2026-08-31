"""Validate a UNSW-MG24 CSV / Parquet drop-in.

Run::

    python -m code.datasets.unsw_mg24.validate_dataset \
        --path code/datasets/unsw_mg24/raw/alerts.csv

Reports:
- file presence and result_source
- inferred timestamp / label columns
- label distribution + DER-class coverage
- count of unmapped labels (collapsed to ``unknown_or_unmapped``)
- a small schema-conformance summary so reviewers know what was loaded

This script is read-only --- it never writes back to disk.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .label_mapping import DER_CLASSES_EXTENDED, to_der_class
from .production_loader import UnswProductionLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="")
    ap.add_argument("--max-samples", type=int, default=10_000)
    args = ap.parse_args()

    loader = UnswProductionLoader(csv_path=args.path or None)
    summary = {
        "available":        loader.available(),
        "path":             loader.path,
        "result_source":    loader.result_source,
        "unavailable_reason": loader.unavailable_reason,
    }
    if not loader.available():
        print(json.dumps(summary, indent=2))
        return

    label_dist: Counter[str] = Counter()
    der_dist: Counter[str] = Counter()
    n = 0
    for sample in loader.iter_samples(max_samples=args.max_samples):
        n += 1
        label_dist[sample.source_label] += 1
        der_dist[sample.der_class] += 1

    summary.update({
        "n_samples":            n,
        "n_distinct_labels":    len(label_dist),
        "label_distribution":   dict(label_dist.most_common()),
        "der_class_distribution": {c: der_dist.get(c, 0) for c in DER_CLASSES_EXTENDED},
        "unknown_or_unmapped_fraction":
            der_dist.get("unknown_or_unmapped", 0) / max(n, 1),
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
