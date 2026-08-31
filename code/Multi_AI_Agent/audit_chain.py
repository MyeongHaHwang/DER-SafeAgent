"""Deterministic hash-chained incident audit log.

Revision note (ijcip_revision_r1r2_20260805): the 2026 submission described a
hash-chained, tamper-evident audit trail, but the chain existed only as a
*prompt instruction* to the Report Agent — the model was asked to echo
``hash_chain_prev`` and nothing computed or verified it. An LLM-echoed hash is
not a tamper-evident construction. The chain is therefore computed and
verified here, deterministically in Python, outside the model's control; the
Report Agent's narrative text remains an LLM product but the integrity fields
do not.

Construction: each record stores ``prev_hash`` and
``record_hash = SHA256(prev_hash || canonical_json(payload))`` where the
canonical form is ``json.dumps(payload, sort_keys=True, separators=(",", ":"))``.
The first record's ``prev_hash`` is the literal ``"GENESIS"``. Verification
recomputes every link, so editing, reordering, inserting, or deleting any
record breaks the chain at that point and at every point after it.

Scope of the claim (kept deliberately narrow): this is an append-only
integrity check that makes silent post-hoc edits detectable to anyone holding
a later record hash. It is not a signature, not a timestamping authority, and
not protection against an attacker who rewrites the whole file and every
downstream copy of the head hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

GENESIS = "GENESIS"


def canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def link_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical(payload)).encode()).hexdigest()


@dataclass
class AuditChain:
    """Append-only chain of incident records."""
    path: Path | None = None
    records: list[dict] = field(default_factory=list)

    @property
    def head(self) -> str:
        return self.records[-1]["record_hash"] if self.records else GENESIS

    def append(self, payload: dict[str, Any]) -> dict:
        rec = {"prev_hash": self.head, "payload": payload}
        rec["record_hash"] = link_hash(rec["prev_hash"], payload)
        self.records.append(rec)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    @classmethod
    def load(cls, path: str | Path) -> "AuditChain":
        p = Path(path)
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return cls(path=p, records=recs)


def verify(records: Iterable[dict]) -> dict:
    """Recompute every link. Returns a report; ``valid`` is True only if the
    whole chain is intact."""
    recs = list(records)
    expected_prev = GENESIS
    for i, rec in enumerate(recs):
        if rec.get("prev_hash") != expected_prev:
            return {"valid": False, "n_records": len(recs), "broken_at": i,
                    "reason": "prev_hash does not match the previous record hash "
                              "(record inserted, deleted, or reordered)"}
        recomputed = link_hash(rec["prev_hash"], rec["payload"])
        if recomputed != rec.get("record_hash"):
            return {"valid": False, "n_records": len(recs), "broken_at": i,
                    "reason": "record_hash does not match its payload "
                              "(payload modified after writing)"}
        expected_prev = rec["record_hash"]
    return {"valid": True, "n_records": len(recs), "broken_at": None,
            "head": expected_prev, "reason": "all links recomputed and matched"}


def verify_file(path: str | Path) -> dict:
    return verify(AuditChain.load(path).records)
