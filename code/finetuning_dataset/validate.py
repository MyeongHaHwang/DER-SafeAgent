"""Standalone validator: re-checks emitted JSONL against the schema.

Run after build_dataset.py to guard against silent corruption / manual edits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from schema import Example

PROCESSED = Path(__file__).parent / "processed"


def validate_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    n = 0
    with path.open() as f:
        for i, line in enumerate(f, 1):
            n += 1
            try:
                Example.model_validate_json(line)
            except Exception as e:
                errors.append(f"{path.name}:{i}: {e}")
    return n, errors


def main() -> int:
    total_errors = 0
    for split in ("train", "val", "test"):
        path = PROCESSED / f"{split}.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        n, errors = validate_file(path)
        print(f"[{split}] {n} rows, {len(errors)} errors")
        for e in errors[:10]:
            print(" ", e)
        total_errors += len(errors)
    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
