from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.utils.dlc_utils import get_bodypart_xy_time
import scripts.db.db_utils as db_utils


def estimate_maze_corners_from_df(
    df: pd.DataFrame,
    *,
    fps: float,
    quantiles: tuple[float, float] = (5.0, 95.0),
    individual: str | None = None,
    likelihood_threshold: float | None = None,
    smoothing_window: int | None = None,
) -> dict:
    """Estimate maze extents (robust corners) from Midback positions.

    Uses `get_bodypart_xy_time()` to extract valid Midback x/y samples and
    returns the requested lower/upper quantiles per axis as the maze corners.
    """
    qlow, qhigh = float(quantiles[0]), float(quantiles[1])
    x, y, _, _, _ = get_bodypart_xy_time(
        df,
        bodypart="Midback",
        fps=float(fps),
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
    )

    if x.size == 0 or y.size == 0:
        raise ValueError("No valid Midback samples available to estimate maze corners.")

    x_min = float(np.nanpercentile(x, qlow))
    x_max = float(np.nanpercentile(x, qhigh))
    y_min = float(np.nanpercentile(y, qlow))
    y_max = float(np.nanpercentile(y, qhigh))

    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])

    return dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, corners=corners)


def estimate_maze_corners_from_id(
    record_id: int,
    *,
    quantiles: tuple[float, float] = (5.0, 95.0),
    individual: str | None = None,
    likelihood_threshold: float | None = None,
    smoothing_window: int | None = None,
) -> dict:
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)
    return estimate_maze_corners_from_df(
        df,
        fps=fps,
        quantiles=quantiles,
        individual=individual,
        likelihood_threshold=likelihood_threshold,
        smoothing_window=smoothing_window,
    )


def normalize_coords(
    coords: np.ndarray,
    corners: dict,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Normalize Nx2 coords into unit square based on corners dict.

    Args:
        coords: Nx2 array of [x,y]
        corners: dict from estimate_maze_corners_from_df
        clip: whether to clip values to [0,1]

    Returns:
        Nx2 array of normalized coordinates in [0,1]
    """
    if coords is None or len(coords) == 0:
        return np.empty((0, 2), dtype=float)

    x_min, x_max = corners["x_min"], corners["x_max"]
    y_min, y_max = corners["y_min"], corners["y_max"]

    dx = x_max - x_min
    dy = y_max - y_min
    eps = 1e-6
    if dx < eps or dy < eps:
        # degenerate box, avoid division by zero
        raise ValueError("Degenerate maze extents (zero width or height); cannot normalize.")

    coords = np.asarray(coords, dtype=float)
    x = (coords[:, 0] - x_min) / dx
    y = (coords[:, 1] - y_min) / dy
    out = np.column_stack([x, y])
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out
