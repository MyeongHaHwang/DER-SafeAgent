"""Metric helpers for the external-benchmark evaluation."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from .label_mapping import DER_CLASSES

# Action vocabulary mirrors `code/simulation/mitigations.py`.
SAFE_ACTIONS: frozenset[str] = frozenset({
    "no_op", "freeze_setpoint", "throttle_ramp", "request_ied_revalidation",
})
IRREVERSIBLE_ACTIONS: frozenset[str] = frozenset({"isolate_inverter"})
ALL_ACTIONS: frozenset[str] = SAFE_ACTIONS | IRREVERSIBLE_ACTIONS

# Expected schema keys for a Hypothesis-Agent-style JSON object.
SCHEMA_REQUIRED: tuple[str, ...] = (
    "attack_class", "affected_asset", "confidence",
    "rationale", "proposed_action", "expected_impact_tier",
)
TIER_VALUES: frozenset[str] = frozenset({
    "negligible", "low", "medium", "high", "severe",
})


# ---------- per-class confusion ----------

def per_class_metrics(rows: Iterable[dict]) -> list[dict]:
    """Compute per-class precision/recall/F1 over the DER taxonomy."""
    rows = list(rows)
    out = []
    for cls in DER_CLASSES:
        tp = fp = fn = 0
        for r in rows:
            true_cls = r["true_class"]
            pred_cls = r["pred_class"]
            if pred_cls == cls and true_cls == cls:
                tp += 1
            elif pred_cls == cls and true_cls != cls:
                fp += 1
            elif pred_cls != cls and true_cls == cls:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append({"class": cls, "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1,
                     "support": tp + fn})
    return out


def macro_f1(per_class: list[dict], exclude_none: bool = True) -> float:
    sub = [r for r in per_class if r["class"] != "none"] if exclude_none else per_class
    return float(sum(r["f1"] for r in sub) / len(sub)) if sub else 0.0


def weighted_f1(per_class: list[dict]) -> float:
    total = sum(r["support"] for r in per_class)
    if total == 0:
        return 0.0
    return float(sum(r["f1"] * r["support"] for r in per_class) / total)


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["pred_class"] == r["true_class"]) / len(rows)


def confusion_matrix(rows: Iterable[dict]) -> list[list[int]]:
    """Return a CxC confusion matrix (rows=true, cols=pred) over DER_CLASSES."""
    idx = {c: i for i, c in enumerate(DER_CLASSES)}
    n = len(DER_CLASSES)
    mat = [[0] * n for _ in range(n)]
    for r in rows:
        i = idx.get(r["true_class"]); j = idx.get(r["pred_class"])
        if i is None or j is None:
            continue
        mat[i][j] += 1
    return mat


# ---------- schema / JSON validity ----------

def is_valid_json(text: Any) -> bool:
    if isinstance(text, dict):
        return True
    if not isinstance(text, str):
        return False
    try:
        obj = json.loads(text)
        return isinstance(obj, dict)
    except Exception:
        return False


def schema_compliant(obj: Any) -> bool:
    if not isinstance(obj, dict):
        try:
            obj = json.loads(obj)
        except Exception:
            return False
    if not isinstance(obj, dict):
        return False
    for k in SCHEMA_REQUIRED:
        if k not in obj:
            return False
    if obj.get("attack_class") not in DER_CLASSES:
        return False
    if obj.get("expected_impact_tier") not in TIER_VALUES:
        return False
    if obj.get("proposed_action") not in ALL_ACTIONS:
        return False
    try:
        c = float(obj.get("confidence", -1))
        if not (0.0 <= c <= 1.0):
            return False
    except (TypeError, ValueError):
        return False
    return True


# ---------- behavioural rates ----------

def safe_action_rate(rows: Iterable[dict]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["pred_action"] in SAFE_ACTIONS) / len(rows)


def forbidden_action_rate(rows: Iterable[dict],
                           forbidden: Iterable[str] | None = None) -> float:
    """Fraction of decisions that propose an action outside the registry."""
    rows = list(rows)
    if not rows:
        return 0.0
    forbidden_set = set(forbidden) if forbidden else set()
    n = 0
    for r in rows:
        a = r["pred_action"]
        if forbidden_set:
            if a in forbidden_set:
                n += 1
        elif a not in ALL_ACTIONS:
            n += 1
    return n / len(rows)


def abstain_rate(rows: Iterable[dict]) -> float:
    """Fraction of decisions that defer to HITL or emit no_op when uncertain."""
    rows = list(rows)
    if not rows:
        return 0.0
    n = sum(1 for r in rows
            if r.get("hitl_required") or r["pred_action"] == "no_op")
    return n / len(rows)


def action_accuracy(rows: Iterable[dict]) -> float:
    """Compare ``pred_action`` to a ``ref_action`` (if present); skip rows
    without a reference."""
    rows = [r for r in rows if r.get("ref_action") is not None]
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["pred_action"] == r["ref_action"]) / len(rows)


# ---------- summary ----------

def summarise(rows: list[dict]) -> dict:
    pcm = per_class_metrics(rows)
    return {
        "n_samples":           len(rows),
        "accuracy":            accuracy(rows),
        "macro_f1":            macro_f1(pcm),
        "weighted_f1":         weighted_f1(pcm),
        "json_validity_rate":  sum(1 for r in rows if r.get("json_valid", True)) / max(len(rows), 1),
        "schema_compliance_rate": sum(1 for r in rows if r.get("schema_compliant", True)) / max(len(rows), 1),
        "safe_action_rate":    safe_action_rate(rows),
        "forbidden_action_rate": forbidden_action_rate(rows),
        "abstain_rate":        abstain_rate(rows),
        "action_accuracy":     action_accuracy(rows),
    }
