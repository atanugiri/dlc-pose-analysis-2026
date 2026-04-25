from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_feature(
    feature_df: pd.DataFrame,
    *,
    feature_name: str,
    how: str = "mean",
    likelihood_min: float | None = None,
) -> float:
    """Summarize a scalar feature column from a per-frame feature DataFrame."""
    if feature_name not in feature_df.columns:
        raise ValueError(
            f"feature_name {feature_name!r} not found in columns: {list(feature_df.columns)}"
        )

    values = feature_df[feature_name].replace([np.inf, -np.inf], np.nan)

    if likelihood_min is not None and "likelihood" in feature_df.columns:
        values = values.where(feature_df["likelihood"] >= float(likelihood_min))

    how_norm = how.strip().lower()

    if how_norm == "mean":
        return float(values.mean(skipna=True))

    if how_norm == "median":
        return float(values.median(skipna=True))

    if how_norm == "max":
        return float(values.max(skipna=True))

    if how_norm == "std":
        return float(values.std(skipna=True))

    raise ValueError("how must be 'mean', 'median', 'max', or 'std'")
