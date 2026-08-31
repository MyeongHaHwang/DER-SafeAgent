"""Multiple-comparison correction (Holm-Bonferroni)."""
from __future__ import annotations

import numpy as np


def holm_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Returns list of {p, p_adj, reject} in original order."""
    arr = np.array(p_values, dtype=float)
    order = np.argsort(arr)
    n = len(arr)
    p_adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * arr[idx]
        running_max = max(running_max, adj)
        p_adj[idx] = min(1.0, running_max)
    return [{"p": float(arr[i]), "p_adj": float(p_adj[i]), "reject": bool(p_adj[i] < alpha)}
            for i in range(n)]
