#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

    dt = 1.0 / float(fps)
    vx = x.diff() / dt
    vy = y.diff() / dt
    speed = np.sqrt(vx * vx + vy * vy)

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


def feature_timeseries_and_value_from_h5(
    h5_path: Path,
    *,
    bodypart: str = "Midback",
    feature_name: str = "speed",
    how: str = "mean",
    likelihood_min: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Return full per-frame DataFrame + one scalar summary value.

    Currently the per-frame DataFrame is the velocity DataFrame.
    """
    vel = _compute_velocity_from_h5(h5_path, bodypart=bodypart, fps=db_utils.DEFAULT_FPS)
    value = feature_summary.summarize_feature(
        vel,
        feature_name=feature_name,
        how=how,
        likelihood_min=likelihood_min,
    )
    return vel, value


def feature_timeseries_and_value_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    feature_name: str = "speed",
    how: str = "mean",
    likelihood_min: float | None = None,
) -> tuple[pd.DataFrame, float, Path]:
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    pose_path = dlc_io.resolve_pose_path(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    vel = _compute_velocity_from_h5(pose_path, bodypart=bodypart, fps=fps)
    value = feature_summary.summarize_feature(
        vel,
        feature_name=feature_name,
        how=how,
        likelihood_min=likelihood_min,
    )
    return vel, value, pose_path


def print_velocity_summary_from_h5(h5_path: Path, *, bodypart: str = "Midback") -> None:
    vel = _compute_velocity_from_h5(h5_path, bodypart=bodypart, fps=db_utils.DEFAULT_FPS)

    print(f"Pose file: {h5_path}")
    print(f"Bodypart: {bodypart}")
    print(f"FPS: {db_utils.DEFAULT_FPS}")
    print(f"Frames: {len(vel)}")
    print(f"Mean speed (px/s): {summarize_speed(vel, how='mean'):.3f}")
    print(f"Median speed (px/s): {summarize_speed(vel, how='median'):.3f}")

    print("\nPreview:")
    with pd.option_context("display.max_columns", 10, "display.width", 140):
        print(vel.head(5))


def print_velocity_summary_from_id(record_id: int, *, bodypart: str = "Midback") -> None:
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    pose_path = dlc_io.resolve_pose_path(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    vel = _compute_velocity_from_h5(pose_path, bodypart=bodypart, fps=fps)

    print(f"ID: {record_id}")
    print(f"Pose file: {pose_path}")
    print(f"Bodypart: {bodypart}")
    print(f"FPS: {fps}")
    print(f"Frames: {len(vel)}")
    print(f"Mean speed (px/s): {summarize_speed(vel, how='mean'):.3f}")
    print(f"Median speed (px/s): {summarize_speed(vel, how='median'):.3f}")

    print("\nPreview:")
    with pd.option_context("display.max_columns", 10, "display.width", 140):
        print(vel.head(5))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame velocity from a DeepLabCut filtered .h5. "
            "Provide either an experimental_metadata id (DB mode) or --h5-path (direct file mode)."
        )
    )
    parser.add_argument("id", type=int, nargs="?", help="experimental_metadata.id (optional if --h5-path provided)")
    parser.add_argument("--h5-path", type=Path, default=None, help="Path to a DLC .h5 (bypasses the database)")
    parser.add_argument("--bodypart", default="Midback", help="Bodypart name (default: Midback)")
    args = parser.parse_args()

    try:
        if args.h5_path is not None:
            if not args.h5_path.exists():
                raise FileNotFoundError(f"File not found: {args.h5_path}")
            print_velocity_summary_from_h5(args.h5_path, bodypart=args.bodypart)
            return

        if args.id is None:
            raise ValueError("Provide either an experimental_metadata id or --h5-path")

        print_velocity_summary_from_id(args.id, bodypart=args.bodypart)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
