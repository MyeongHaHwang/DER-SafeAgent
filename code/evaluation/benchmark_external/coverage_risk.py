"""Selective-prediction metrics for the external-benchmark suite.

When the detector is allowed to abstain (e.g. by emitting ``no_op`` or
routing to HITL), the headline accuracy/macro-F1 numbers must be paired
with a ``coverage`` metric so that a high-abstaining detector cannot
appear to dominate by silence alone.

This module computes the classical selective-prediction quantities:

* coverage   --- fraction of samples on which the detector commits to a
                  non-abstention prediction
* selective_risk --- error rate restricted to the covered subset
* coverage--risk curve --- coverage and selective-risk at a sweep of
                            abstention thresholds
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def coverage_and_selective_risk(rows: list[dict]) -> dict[str, float]:
    """Single-point coverage / selective-risk pair.

    A row is considered an abstention when ``hitl_required`` is True
    OR when the predicted action is ``no_op``. Risk is the
    misclassification rate on the *covered* subset.
    """
    n = len(rows)
    if n == 0:
        return {"coverage": 0.0, "selective_risk": float("nan"),
                 "n_covered": 0, "n_abstained": 0}
    covered = [r for r in rows
                 if not r.get("hitl_required") and r.get("pred_action") != "no_op"]
    n_cov = len(covered)
    if n_cov == 0:
        return {"coverage": 0.0, "selective_risk": float("nan"),
                 "n_covered": 0, "n_abstained": n}
    err = sum(1 for r in covered if r["pred_class"] != r["true_class"])
    return {
        "coverage":         n_cov / n,
        "selective_risk":   err / n_cov,
        "n_covered":        n_cov,
        "n_abstained":      n - n_cov,
    }


def coverage_risk_curve(rows: list[dict],
                          thresholds: Iterable[float] | None = None) -> list[dict]:
    """Sweep an abstention threshold over the predicted confidence.

    At threshold ``t``, a sample is *covered* iff its predicted
    confidence is at least ``t`` AND its predicted action is not
    ``no_op`` AND HITL is not required. Useful for plotting a curve.
    """
    if thresholds is None:
        thresholds = list(np.linspace(0.0, 1.0, 11))
    out = []
    for t in thresholds:
        covered = [r for r in rows
                     if (not r.get("hitl_required"))
                     and r.get("pred_action") != "no_op"
                     and float(r.get("confidence", 0.0)) >= float(t)]
        n = len(rows)
        n_cov = len(covered)
        if n_cov == 0:
            out.append({"threshold": float(t),
                          "coverage": 0.0, "selective_risk": float("nan"),
                          "n_covered": 0, "n_abstained": n})
            continue
        err = sum(1 for r in covered if r["pred_class"] != r["true_class"])
        out.append({"threshold": float(t),
                      "coverage":      n_cov / n,
                      "selective_risk": err / n_cov,
                      "n_covered":     n_cov,
                      "n_abstained":   n - n_cov})
    return out


def correct_abstention_rate(rows: list[dict]) -> float:
    """Fraction of rows where the *correct* behaviour was to abstain
    (true class is ``unknown_or_unmapped`` or label is uncertain) AND
    the detector did abstain."""
    if not rows:
        return 0.0
    target = [r for r in rows
                if r["true_class"] in {"unknown_or_unmapped", "none"}]
    if not target:
        return 0.0
    correct = sum(1 for r in target
                    if r.get("hitl_required") or r.get("pred_action") == "no_op")
    return correct / len(target)


def unsafe_forced_action_rate(rows: list[dict]) -> float:
    """Fraction of rows where the detector emitted an action despite
    the true class being unmapped --- a key trustworthy-AI failure mode
    on packet-level external benchmarks."""
    if not rows:
        return 0.0
    target = [r for r in rows if r["true_class"] == "unknown_or_unmapped"]
    if not target:
        return 0.0
    forced = sum(1 for r in target
                  if r.get("pred_action") not in {"no_op", None}
                  and not r.get("hitl_required"))
    return forced / len(target)
