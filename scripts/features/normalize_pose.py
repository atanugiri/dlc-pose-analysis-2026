from __future__ import annotations

import numpy as np

from scripts.utils.dlc_utils import get_bodypart_xy_time
import scripts.db.db_utils as db_utils


def normalize_coords(
    coords: np.ndarray,
    corners: dict,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Normalize Nx2 coords into the unit square given a corners dict.

    Args:
        coords: Nx2 array of [x, y]
        corners: dict with keys x_min, x_max, y_min, y_max
        clip: whether to clip values to [0, 1]

    Returns:
        Nx2 array of normalized coordinates in [0, 1]
    """
    if coords is None or len(coords) == 0:
        return np.empty((0, 2), dtype=float)

    x_min, x_max = corners["x_min"], corners["x_max"]
    y_min, y_max = corners["y_min"], corners["y_max"]

    dx = x_max - x_min
    dy = y_max - y_min
    eps = 1e-6
    if dx < eps or dy < eps:
        raise ValueError("Degenerate maze extents (zero width or height); cannot normalize.")

    coords = np.asarray(coords, dtype=float)
    x = (coords[:, 0] - x_min) / dx
    y = (coords[:, 1] - y_min) / dy
    out = np.column_stack([x, y])
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def get_bodypart_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    likelihood_threshold: float | None = 0.9,
    smoothing_window: int | None = None,
) -> tuple:
    """Return raw (x, y, likelihood, time, index) for one bodypart from DB record.

    Loads the filtered pose file for the record and extracts coordinates
    without normalization.
    """
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    return get_bodypart_xy_time(
        df,
        bodypart=bodypart,
        fps=float(fps),
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
    )


def normalize_bodypart_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    likelihood_threshold: float | None = 0.9,
    smoothing_window: int | None = None,
) -> tuple:
    """Return normalized (x, y, likelihood, time, index) for one bodypart.

    Coordinates are normalized using cached corners in DB.
    """
    x, y, likelihood, time, index = get_bodypart_from_id(
        record_id,
        bodypart=bodypart,
        individual=individual,
        likelihood_threshold=likelihood_threshold,
        smoothing_window=smoothing_window,
    )

    if x.size > 0:
        corners = db_utils.get_cached_maze_corners(record_id)
        if corners is None:
            raise ValueError(
                f"No cached maze_corners found for ID {record_id}. "
                "Run scripts.db.inject_maze_corners first."
            )
        coords = np.column_stack([x, y])
        coords_norm = normalize_coords(coords, corners, clip=True)
        x = coords_norm[:, 0]
        y = coords_norm[:, 1]

    return x, y, likelihood, time, index
