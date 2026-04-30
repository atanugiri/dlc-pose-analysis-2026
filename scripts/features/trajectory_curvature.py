from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.db.db_utils as db_utils
import scripts.utils.dlc_utils as dlc_utils
import scripts.features.feature_summary as feature_summary


def compute_curvature_from_df(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
) -> pd.DataFrame:
    """Compute per-frame curvature for one DLC bodypart.
    
    Curvature is computed as κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2),
    where primes denote derivatives with respect to time.
    
    If `speed_thresh` is provided, curvature is set to 0 for frames where
    speed is below this threshold, as low-speed curvature can be unreliable.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    x, y, likelihood, time, index = dlc_utils.get_bodypart_xy_time(
        df, bodypart=bodypart, fps=fps, individual=individual, smoothing_window=smoothing_window
    )

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
        curvature = np.where(speed < speed_thresh, 0.0, curvature) # Probably should be np.nan

    out = pd.DataFrame({
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

    return out


def compute_curvature_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
) -> pd.DataFrame:
    """Load one DB record and compute per-frame curvature."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    return compute_curvature_from_df(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        smoothing_window=smoothing_window,
        speed_thresh=speed_thresh,
    )


def summarize_curvature(
    curvature_df: pd.DataFrame,
    *,
    how: str = "mean",
    likelihood_min: float | None = None,
) -> float:
    """Summarize curvature from a per-frame curvature DataFrame.
    
    Curvature should already be thresholded in `compute_curvature_from_df`.
    `speed_thresh` is kept for API consistency.
    """
    return feature_summary.summarize_feature(
        curvature_df,
        feature_name="curvature",
        how=how,
        likelihood_min=likelihood_min,
    )


def summarize_curvature_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    likelihood_min: float | None = None,
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
) -> float:
    """Compute one scalar curvature summary for one DB record id."""
    curvature_df = compute_curvature_from_id(
        record_id,
        bodypart=bodypart,
        individual=individual,
        smoothing_window=smoothing_window,
        speed_thresh=speed_thresh,
    )

    return summarize_curvature(
        curvature_df,
        how=how,
        likelihood_min=likelihood_min,
        speed_thresh=speed_thresh,
    )


def summarize_curvature_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    likelihood_min: float | None = None,
    individual: str | None = None,
    smoothing_window: int = 5,
    speed_thresh: float = 0.01,
) -> list[float]:
    """Compute one scalar curvature summary per record id."""
    return [
        summarize_curvature_from_id(
            record_id,
            bodypart=bodypart,
            how=how,
            likelihood_min=likelihood_min,
            individual=individual,
            smoothing_window=smoothing_window,
            speed_thresh=speed_thresh,
        )
        for record_id in record_ids
    ]
