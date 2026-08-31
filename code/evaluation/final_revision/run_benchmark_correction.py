"""Public-benchmark correction (brief §14).

The published figure of 27,533 UNSW-MG24 samples is the concatenation of the
official training (19,274) and test (8,261) splits. No learning happens on that
data — the mapping is a fixed label table — so this is not train-on-test
leakage in the usual sense, but it is also not an official test-split
generalisation number and must not be described as one.

This driver re-runs the label mapping on each split separately so the
manuscript can report the official test split, and it states plainly which
artifact is being evaluated: a **deterministic label mapper**, not an LLM and
not the complete response system.

Run: python3 -m code.evaluation.final_revision.run_benchmark_correction
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

TAG = "ijcip_final_revision"
OUT = Path("code/results") / TAG / "benchmark_correction"
DATA = Path("code/datasets/unsw_mg24")
SPLITS = {"train": DATA / "training-flow.csv",
          "test": DATA / "test-flow.csv",
          "merged_as_published": DATA / "full-flow.csv"}

DER_CLASSES = {"none", "fdi", "replay", "command_spoof", "dos", "firmware"}


def _label_col(header: list[str]) -> str | None:
    for c in ("attack_name", "attack_step", "attack_flag", "label", "Label",
              "attack", "Attack", "attack_type", "class"):
        if c in header:
            return c
    return None


def map_label(raw: str) -> str:
    """Fixed, deterministic mapping into the six-class DER taxonomy.

    Anything without a confident cyber-physical equivalent is routed to
    ``unknown_or_unmapped``, on which the system is expected to abstain.
    """
    s = (raw or "").strip().lower()
    if s in ("0", "benign", "normal", "none", ""):
        return "none"
    if "dos" in s or "flood" in s:
        return "dos"
    if "replay" in s:
        return "replay"
    if "inject" in s or "fdi" in s or "false" in s:
        return "fdi"
    if "spoof" in s or "mitm" in s or "command" in s:
        return "command_spoof"
    if "firmware" in s or "malware" in s:
        return "firmware"
    return "unknown_or_unmapped"


def evaluate(path: Path) -> dict:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        col = _label_col(header)
        if col is None:
            return {"file": str(path), "error": "no label column found",
                    "header_sample": header[:12]}
        raw_counts: Counter = Counter()
        mapped_counts: Counter = Counter()
        n = 0
        for row in reader:
            n += 1
            raw = row.get(col, "")
            raw_counts[raw] += 1
            mapped_counts[map_label(raw)] += 1
    n_unmapped = mapped_counts.get("unknown_or_unmapped", 0)
    covered = n - n_unmapped
    return {
        "file": str(path), "label_column": col, "n_samples": n,
        "coverage": covered / n if n else 0.0,
        "abstain_rate": n_unmapped / n if n else 0.0,
        "mapped_distribution": dict(mapped_counts),
        "n_distinct_raw_labels": len(raw_counts),
        "top_raw_labels": dict(raw_counts.most_common(8)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {k: evaluate(p) for k, p in SPLITS.items() if p.exists()}
    missing = [k for k, p in SPLITS.items() if not p.exists()]

    report = {
        "artifact_evaluated": ("deterministic label mapper (fixed table). "
                               "NOT an LLM, NOT the complete response system, "
                               "and no learning is performed on this data."),
        "published_figure_explained": (
            "The manuscript's 27,533 samples equal training (19,274) + test "
            "(8,261) concatenated. Reported here per split so the official "
            "test split can be cited on its own."),
        "splits": results,
        "missing_splits": missing,
        "caveat": ("Because the mapper is a fixed table and never sees a "
                   "label at inference time, split choice does not create "
                   "optimistic bias; the correction is about accurate "
                   "description, not about a leaked score."),
    }
    (OUT / "benchmark_split_correction.json").write_text(json.dumps(report, indent=2))

    rows = []
    for k, r in results.items():
        if "error" in r:
            rows.append({"split": k, "n_samples": None, "note": r["error"]})
            continue
        rows.append({"split": k, "n_samples": r["n_samples"],
                     "coverage": round(r["coverage"], 4),
                     "abstain_rate": round(r["abstain_rate"], 4),
                     "n_distinct_raw_labels": r["n_distinct_raw_labels"]})
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "benchmark_split_correction.csv", index=False)
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk in ("n_samples", "coverage", "abstain_rate",
                                    "label_column", "error")}
                      for k, v in results.items()}, indent=2))
    if missing:
        print("missing splits:", missing)


if __name__ == "__main__":
    main()
