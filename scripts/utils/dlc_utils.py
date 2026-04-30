from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def get_bodypart_xy_time(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Extract valid x, y, likelihood, time, and index for one bodypart.

    If provided, `smoothing_window` applies a moving-average smoothing
    (`uniform_filter1d`, nearest-edge padding) to the x/y trajectory before
    filtering valid samples.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC-style MultiIndex columns.")

    scorer = df.columns.get_level_values(0).unique().tolist()[0]

    if df.columns.nlevels >= 4:
        if individual is None:
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(
                "Multi-animal data requires `individual`. "
                f"Available individuals: {available}"
            )

        if individual not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Individual {individual!r} not found. Available: {available}")

        coords = df[scorer][individual][bodypart]
    else:
        if bodypart not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

        coords = df[scorer][bodypart]

    if not {"x", "y", "likelihood"}.issubset(coords.columns):
        raise ValueError(
            f"Bodypart {bodypart!r} is missing x/y/likelihood columns. Found: {list(coords.columns)}"
        )

    x = coords["x"].astype(float).to_numpy()
    y = coords["y"].astype(float).to_numpy()
    likelihood = coords["likelihood"].astype(float).to_numpy()

    # Set low-likelihood x/y to NaN and interpolate when thresholding is enabled
    if likelihood_threshold is not None:
        thr = float(likelihood_threshold)
        low_mask = likelihood < thr
        if low_mask.any():
            x = x.copy()
            y = y.copy()
            x[low_mask] = np.nan
            y[low_mask] = np.nan
            # interpolate NaNs in x/y in both directions
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    if smoothing_window is not None:
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be a positive integer or None.")
        w = int(smoothing_window)
        if w > 1:
            if w % 2 == 0:
                w += 1
            w = max(3, w)
            x = uniform_filter1d(x, size=w, mode="nearest")
            y = uniform_filter1d(y, size=w, mode="nearest")

    frame_number = np.arange(len(coords))
    time = frame_number / fps

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood)

    return x[valid], y[valid], likelihood[valid], time[valid], coords.index[valid]

def main():
    from scripts.db import db_utils

    id = 123  # Example record ID
    filtered_pose_file = db_utils.get_filtered_pose_file(id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(id)
    x, y, likelihood, time, index = get_bodypart_xy_time(
        df,
        bodypart="Midback",
        fps=fps,
    )
    print(f"x: {x[:5]}, y: {y[:5]}, likelihood: {likelihood[:5]}, time: {time[:5]}, index: {index[:5]}")

if __name__ == "__main__":
    main()