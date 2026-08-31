"""Portability helpers for frozen manifests.

The hash-frozen configuration manifests (e.g.
``code/configs/ijcip_final_v3/holdout_v3_manifest.csv``) record the absolute
``config_path`` of each scenario as it existed on the machine that froze the
holdout.  Rewriting those CSVs would invalidate their recorded SHA-256
digests, so the release keeps the frozen bytes and remaps the paths at load
time instead: if the recorded path does not exist, the ``code/simulation/…``
suffix is resolved relative to the repository root (the current working
directory, since all drivers run as ``python3 -m code.…`` from the root).
"""

from __future__ import annotations

from pathlib import Path

_MARKER = "code/simulation/"


def resolve_config_path(recorded: str | Path) -> Path:
    """Resolve a manifest-recorded scenario path on this machine."""
    p = Path(recorded)
    if p.exists():
        return p
    s = str(recorded)
    idx = s.find(_MARKER)
    if idx != -1:
        rel = Path(s[idx:])
        if rel.exists():
            return rel
    raise FileNotFoundError(
        f"scenario config not found: {recorded!r} (also tried repo-relative "
        f"remap; run drivers from the repository root)"
    )
