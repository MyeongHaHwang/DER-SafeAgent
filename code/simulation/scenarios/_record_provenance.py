"""Record provenance metadata for a `feeder.dss` file.

Writes a sibling `feeder_provenance.json` containing:
  - sha256 of the .dss file
  - line count
  - first 10 non-comment lines (for human spot-check)
  - timestamp

Reviewers re-running the pipeline can compare hashes to confirm that the
scenario is using the same feeder definition as the paper.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _record_provenance.py <path/to/feeder.dss>")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"file not found: {p}")
        return 1

    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    non_comment = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("!")]

    meta = {
        "path": str(p),
        "sha256": h,
        "line_count": text.count("\n") + 1,
        "non_comment_preview": non_comment[:10],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    out = p.with_name("feeder_provenance.json")
    out.write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
