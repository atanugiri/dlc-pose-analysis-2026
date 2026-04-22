from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.motion_features import _compute_velocity_from_h5


def summarize_feature(
    feature_df: pd.DataFrame,
    *,
    feature_name: str,
    how: str = "mean",
    likelihood_min: float | None = None,
) -> float:
    """Summarize a scalar feature column from a per-frame feature DataFrame.

    Parameters
    ----------
    feature_df:
        Per-frame features (one row per frame).
    feature_name:
        Column name to summarize (e.g. 'speed').
    how:
        Aggregation: 'mean' or 'median'.
    likelihood_min:
        If provided and a 'likelihood' column exists, frames with likelihood < threshold are ignored.

    Returns
    -------
    float
        The requested summary statistic.
    """
    if feature_name not in feature_df.columns:
        raise ValueError(f"feature_name {feature_name!r} not found in columns: {list(feature_df.columns)}")

    values = feature_df[feature_name].replace([np.inf, -np.inf], np.nan)

    if likelihood_min is not None and "likelihood" in feature_df.columns:
        values = values.where(feature_df["likelihood"] >= float(likelihood_min))

    how_norm = how.strip().lower()
    if how_norm == "mean":
        return float(values.mean(skipna=True))
    if how_norm == "median":
        return float(values.median(skipna=True))
    raise ValueError("how must be 'mean' or 'median'")

if __name__ == "__main__":
        import argparse
        import db_utils
        import motion_features

        parser = argparse.ArgumentParser()
        parser.add_argument("record_id", type=int)
        args = parser.parse_args()

        filtered_pose_file = db_utils.get_filtered_pose_file(args.record_id)
        df = motion_features._compute_velocity_from_h5(filtered_pose_file)
        print(df.head())