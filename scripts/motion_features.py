#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Support running as a script (preferred): `python scripts/motion_features.py ...`
# and importing in notebooks: `from scripts.motion_features import ...`
try:
    import db_utils
    import dlc_io
    import feature_summary
except Exception:  # pragma: no cover
    from . import db_utils, dlc_io, feature_summary  # type: ignore


def _compute_velocity_from_h5(
    h5_path: Path,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
) -> pd.DataFrame:
    """Compute per-frame (vx, vy, speed) for one bodypart from a DLC .h5."""
    if fps <= 0:
        raise ValueError("fps must be > 0")

    df = dlc_io.load_dlc_dataframe(h5_path)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
        raise ValueError(
            "Expected DLC-style MultiIndex columns (scorer, bodypart, coord). "
            f"Got: {type(df.columns).__name__}"
        )

    scorer = df.columns.get_level_values(0).unique().tolist()[0]
    if bodypart not in df[scorer].columns.get_level_values(0):
        available = df[scorer].columns.get_level_values(0).unique().tolist()
        raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

    coords = df[scorer][bodypart]
    if not {"x", "y"}.issubset(set(coords.columns)):
        raise ValueError(f"Bodypart {bodypart!r} is missing x/y columns. Found: {list(coords.columns)}")

    x = coords["x"].astype(float)
    y = coords["y"].astype(float)

    fps_f = float(fps)
    vx = x.diff() * fps_f
    vy = y.diff() * fps_f
    speed = np.hypot(vx, vy)

    out = pd.DataFrame({"x": x, "y": y, "vx": vx, "vy": vy, "speed": speed})
    if "likelihood" in coords.columns:
        out["likelihood"] = coords["likelihood"].astype(float)
    return out


def compute_velocity_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
) -> pd.DataFrame:
    """Compute per-frame velocity for a DB record id.

    Uses:
    - db_utils.get_filtered_pose_file(id)
    - db_utils.get_fps(id) (falls back to 15 until you populate frame_rate)
    - dlc_io.resolve_pose_path(...)
    """
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    pose_path = dlc_io.resolve_pose_path(filtered_pose_file)
    fps = db_utils.get_fps(record_id)
    return _compute_velocity_from_h5(pose_path, bodypart=bodypart, fps=fps)


def summarize_speed(
    velocity_df: pd.DataFrame,
    *,
    how: str = "mean",
    likelihood_min: float | None = None,
) -> float:
    return feature_summary.summarize_feature(
        velocity_df,
        feature_name="speed",
        how=how,
        likelihood_min=likelihood_min,
    )


def summarize_speed_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    how: str = "mean",
    likelihood_min: float | None = None,
) -> list[float]:
    """Compute one scalar speed summary per record id.

    Returns values in the same order as `record_ids`.
    """
    return [
        summarize_speed(
            compute_velocity_from_id(record_id, bodypart=bodypart),
            how=how,
            likelihood_min=likelihood_min,
        )
        for record_id in record_ids
    ]
