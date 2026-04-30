from __future__ import annotations

import argparse

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
) -> pd.DataFrame:
    """Compute per-frame x, y, vx, vy, and speed for one DLC bodypart."""
    if fps <= 0:
        raise ValueError("fps must be > 0")

    x, y, likelihood, time, index = dlc_utils.get_bodypart_xy_time(
        df, bodypart=bodypart, fps=fps, individual=individual, smoothing_window=smoothing_window, likelihood_threshold=likelihood_threshold
    )

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
) -> pd.DataFrame:
    """Load one DB record and compute per-frame velocity."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    return compute_velocity_from_df(df, bodypart=bodypart, fps=fps, individual=individual, smoothing_window=smoothing_window, likelihood_threshold=likelihood_threshold)


def summarize_speed(
    velocity_df: pd.DataFrame,
    *,
    how: str = "mean",
    likelihood_min: float | None = None,
) -> float:
    """Summarize speed from a per-frame velocity DataFrame."""
    return feature_summary.summarize_feature(
        velocity_df,
        feature_name="speed",
        how=how,
        likelihood_min=likelihood_min,
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
) -> float:
    """Compute one scalar speed summary for one DB record id."""
    velocity_df = compute_velocity_from_id(
        record_id, bodypart=bodypart, individual=individual, smoothing_window=smoothing_window, likelihood_threshold=likelihood_threshold
    )

    return summarize_speed(
        velocity_df,
        how=how,
        likelihood_min=likelihood_min,
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
        )
        for record_id in record_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute summarized speed values for one or more record IDs."
    )
    parser.add_argument(
        "record_ids",
        nargs="+",
        type=int,
        help="One or more database record IDs.",
    )
    parser.add_argument(
        "--bodypart",
        default="Midback",
        help="Bodypart name to use for velocity/speed computation.",
    )
    parser.add_argument(
        "--how",
        default="mean",
        choices=["mean", "median", "max", "std"],
        help="Summary method.",
    )
    parser.add_argument(
        "--likelihood-min",
        type=float,
        default=None,
        help="Optional minimum likelihood threshold.",
    )

    args = parser.parse_args()

    values = summarize_speed_from_ids(
        args.record_ids,
        bodypart=args.bodypart,
        how=args.how,
        likelihood_min=args.likelihood_min,
    )

    for record_id, value in zip(args.record_ids, values):
        print(f"{record_id}\t{value}")


if __name__ == "__main__":
    main()