"""Build the fine-tuning corpus from raw sources into validated JSONL.

Stages
------
1. Load raw shards from `raw/<source>/*.json`.
2. Normalize to `Example` (schema.py).
3. Validate; reject invalid rows into `raw/REJECTED.md`.
4. Stratified split by (scenario, attack_class).
5. Emit `processed/{train,val,test}.jsonl` + `processed/manifest.json` (row hashes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from schema import Example

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
SPLITS = (("train", 0.8), ("val", 0.1), ("test", 0.1))


def load_raw() -> list[Example]:
    examples: list[Example] = []
    rejected: list[tuple[str, str]] = []
    for shard in sorted(RAW.glob("*/*.json")):
        for row in json.loads(shard.read_text()):
            try:
                examples.append(Example.model_validate(row))
            except Exception as e:
                rejected.append((shard.name, str(e)))
    if rejected:
        (RAW / "REJECTED.md").write_text(
            "# Rejected rows\n\n" + "\n".join(f"- `{f}`: {e}" for f, e in rejected)
        )
    return examples


def stratified_split(examples: list[Example], seed: int,
                     min_per_split: int = 1) -> dict[str, list[Example]]:
    """Stratified by (scenario, attack_class).

    With small buckets, naive `round(n * frac)` rounds val/test to zero. We
    enforce a minimum of `min_per_split` per non-train split *globally* (not
    per-bucket), which is sufficient for early-stage pipeline validation.
    """
    buckets: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for ex in examples:
        buckets[(ex.scenario, ex.attack_class.value)].append(ex)

    rng = random.Random(seed)
    out: dict[str, list[Example]] = {name: [] for name, _ in SPLITS}
    for items in buckets.values():
        rng.shuffle(items)
        n = len(items)
        idx = 0
        for name, frac in SPLITS:
            take = round(n * frac) if name != SPLITS[-1][0] else n - idx
            out[name].extend(items[idx:idx + take])
            idx += take

    # Globally guarantee min_per_split for val/test by stealing from train if
    # needed. We pop the most-populated train rows so we don't re-bias one
    # bucket disproportionately.
    train_name = SPLITS[0][0]
    for name, _ in SPLITS[1:]:
        deficit = min_per_split - len(out[name])
        while deficit > 0 and out[train_name]:
            out[name].append(out[train_name].pop(0))
            deficit -= 1
    return out


def write_split(name: str, items: list[Example]) -> dict[str, str]:
    PROCESSED.mkdir(exist_ok=True)
    path = PROCESSED / f"{name}.jsonl"
    hashes: dict[str, str] = {}
    with path.open("w") as f:
        for ex in items:
            line = ex.model_dump_json()
            f.write(line + "\n")
            hashes[ex.id] = hashlib.sha256(line.encode()).hexdigest()[:16]
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-per-split", type=int, default=1,
                        help="Global minimum rows in val/test (defends against tiny pilot data)")
    args = parser.parse_args()

    examples = load_raw()
    splits = stratified_split(examples, args.seed, min_per_split=args.min_per_split)

    manifest = {"seed": args.seed, "splits": {}}
    for name, items in splits.items():
        manifest["splits"][name] = {
            "n": len(items),
            "hashes": write_split(name, items),
        }
    (PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print({k: v["n"] for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
