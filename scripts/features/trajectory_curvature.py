from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.features.feature_summary as feature_summary
from scripts.features.normalize_pose import (
    normalize_bodypart_from_id,
    get_bodypart_from_id,
)


def compute_curvature_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
    likelihood_threshold: float | None = 0.9,
    normalization: bool = True,
) -> pd.DataFrame:
    """Load one DB record and compute per-frame curvature."""
    if normalization:
        x, y, likelihood, time, index = normalize_bodypart_from_id(
            record_id,
            bodypart=bodypart,
            individual=individual,
            likelihood_threshold=likelihood_threshold,
            smoothing_window=smoothing_window,
        )
    else:
        x, y, likelihood, time, index = get_bodypart_from_id(
            record_id,
            bodypart=bodypart,
            individual=individual,
            likelihood_threshold=likelihood_threshold,
            smoothing_window=smoothing_window,
    )

    # Convert arrays to Series with proper index
    x = pd.Series(x, index=index, dtype=float)
    y = pd.Series(y, index=index, dtype=float)
    likelihood = pd.Series(likelihood, index=index, dtype=float)
    t = pd.Series(time, index=index).astype(float)

    dt = t.diff().replace(0, np.nan)
    
    # First derivatives (velocities)
    vx = x.diff() / dt
    vy = y.diff() / dt
    
    # Second derivatives (accelerations)
    ax = vx.diff() / dt
    ay = vy.diff() / dt
    
    # Speed for threshold filtering
    speed = np.hypot(vx, vy)
    
    # Curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(vx * ay - vy * ax)
    denominator = (vx**2 + vy**2)**(3/2)
    
    # Avoid division by zero and invalid operations
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.where(denominator > 0, numerator / denominator, np.nan)
    
    # Apply speed threshold if provided
    if speed_thresh is not None and speed_thresh > 0:
        curvature = np.where(speed < speed_thresh, np.nan, curvature)
    
    return pd.DataFrame({
        "time": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "speed": speed,
        "curvature": curvature,
        "likelihood": likelihood,
    })


def summarize_curvature(
    curvature_df: pd.DataFrame,
    *,
    how: str = "mean",
) -> float:
    """Summarize curvature from a per-frame curvature DataFrame.
    
    Curvature should already be thresholded in `compute_curvature_from_df`.
    `speed_thresh` is kept for API consistency.
    """
    return feature_summary.summarize_feature(
        curvature_df,
        feature_name="curvature",
        how=how,
    )


def summarize_curvature_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
    likelihood_threshold: float | None = 0.9,
) -> float:
    """Compute one scalar curvature summary for one DB record id."""
    curvature_df = compute_curvature_from_id(
        record_id,
        bodypart=bodypart,
        individual=individual,
        smoothing_window=smoothing_window,
        speed_thresh=speed_thresh,
        likelihood_threshold=likelihood_threshold,
    )

    return summarize_curvature(
        curvature_df,
        how=how,
    )


def summarize_curvature_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
    likelihood_threshold: float | None = 0.9,
) -> list[float]:
    """Compute one scalar curvature summary per record id."""
    return [
        summarize_curvature_from_id(
            record_id,
            bodypart=bodypart,
            how=how,
            individual=individual,
            smoothing_window=smoothing_window,
            speed_thresh=speed_thresh,
            likelihood_threshold=likelihood_threshold,
        )
        for record_id in record_ids
    ]
