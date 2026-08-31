"""Aggregate per-seed runs into method-level summaries."""
from __future__ import annotations

import pandas as pd


def aggregate_seeds(df: pd.DataFrame, metric: str,
                    group_keys: tuple[str, ...] = ("scenario", "detector")) -> pd.DataFrame:
    """Mean / median / IQR of `metric` across seeds for each (scenario, detector)."""
    g = df.groupby(list(group_keys))[metric]
    out = g.agg(mean="mean", median="median", q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75), n="count").reset_index()
    out["iqr"] = out["q75"] - out["q25"]
    return out
