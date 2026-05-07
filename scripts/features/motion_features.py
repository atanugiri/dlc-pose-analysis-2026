from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.db.db_utils as db_utils
import scripts.utils.dlc_utils as dlc_utils
import scripts.features.feature_summary as feature_summary


def compute_velocity_from_df(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    corners: dict | None = None,
) -> pd.DataFrame:
    """Compute per-frame x, y, vx, vy, and speed for one DLC bodypart.

    If `corners` is provided the x/y coordinates will be normalized into the
    unit square using `scripts.features.normalize_maze.normalize_coords`
    before velocities are computed.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    x, y, likelihood, time, index = dlc_utils.get_bodypart_xy_time(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
    )

    # apply normalization when corners provided
    if corners is not None:
        try:
            from scripts.features.normalize_maze import normalize_coords

            coords = np.column_stack([x, y])
            coords_norm = normalize_coords(coords, corners, clip=True)
            x = coords_norm[:, 0]
            y = coords_norm[:, 1]
        except Exception:
            # if normalization fails, fall back to raw coords
            pass

    x = pd.Series(x, index=index, dtype=float)
    y = pd.Series(y, index=index, dtype=float)
    likelihood = pd.Series(likelihood, index=index, dtype=float)
    t = pd.Series(time, index=index).astype(float)

    dt = t.diff().replace(0, np.nan)
    vx = x.diff() / dt
    vy = y.diff() / dt

    speed = np.hypot(vx, vy)

    out = pd.DataFrame({"time": t, "x": x, "y": y, "vx": vx, "vy": vy, "speed": speed, "likelihood": likelihood})

    return out


def compute_velocity_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    normalization: bool = True,
) -> pd.DataFrame:
    """Load one DB record and compute per-frame velocity."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    corners = None
    if normalization:
        # determine whether to pool across maze_number for this record
        pool = False
        try:
            conn = db_utils.connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT maze_number FROM public.experimental_metadata WHERE id = %s",
                        (record_id,),
                    )
                    row = cur.fetchone()
                    pool = bool(row and row[0] is not None)
            finally:
                conn.close()
        except Exception:
            pool = False

        try:
            if pool:
                from scripts.features.normalize_maze import estimate_maze_corners_from_group as _est
            else:
                from scripts.features.normalize_maze import estimate_maze_corners_from_id as _est

            corners = _est(
                record_id,
                individual=individual,
                likelihood_threshold=likelihood_threshold,
                smoothing_window=smoothing_window,
            )
        except Exception:
            corners = None

    return compute_velocity_from_df(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
        corners=corners,
    )


def summarize_speed(
    velocity_df: pd.DataFrame,
    *,
    how: str = "mean",
) -> float:
    """Summarize speed from a per-frame velocity DataFrame."""
    return feature_summary.summarize_feature(
        velocity_df,
        feature_name="speed",
        how=how,
    )


def summarize_speed_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    likelihood_min: float | None = None,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    normalization: bool = True,
) -> float:
    """Compute one scalar speed summary for one DB record id."""
    velocity_df = compute_velocity_from_id(
        record_id,
        bodypart=bodypart,
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
        normalization=normalization,
    )

    return summarize_speed(
        velocity_df,
        how=how,
    )


def summarize_speed_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    likelihood_min: float | None = None,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    normalization: bool = True,
) -> list[float]:
    """Compute one scalar speed summary per record id."""
    return [
        summarize_speed_from_id(
            record_id,
            bodypart=bodypart,
            how=how,
            likelihood_min=likelihood_min,
            individual=individual,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
            normalization=normalization,
        )
        for record_id in record_ids
    ]



if __name__ == "__main__":
    record_ids = [1, 2]
    values = summarize_speed_from_ids(record_ids, bodypart="Midback", how="mean", likelihood_threshold=0.5, normalization=True)
    print(values)
