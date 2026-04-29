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
) -> pd.DataFrame:
    """Compute per-frame x, y, vx, vy, and speed for one DLC bodypart."""
    x, y, likelihood, time = dlc_utils.get_bodypart_xy_time(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
    )

    x = pd.Series(x, index=df.index).astype(float)
    y = pd.Series(y, index=df.index).astype(float)
    likelihood = pd.Series(likelihood, index=df.index).astype(float)

    if time is not None:
        t = pd.Series(time, index=df.index).astype(float)
        dt = t.diff().replace(0, np.nan)
        vx = x.diff() / dt
        vy = y.diff() / dt
    else:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        vx = x.diff() * float(fps)
        vy = y.diff() * float(fps)

    speed = np.hypot(vx, vy)

    out = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "speed": speed,
        }
    )
    # Include likelihood (from dlc_utils.get_bodypart_xy_time) and time
    # if available so downstream code can use them directly.
    out["likelihood"] = likelihood
    if time is not None:
        out["time"] = t
    return out


def compute_velocity_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
) -> pd.DataFrame:
    """Load one DB record and compute per-frame velocity."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    return compute_velocity_from_df(df, bodypart=bodypart, fps=fps, individual=individual)


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
) -> float:
    """Compute one scalar speed summary for one DB record id."""
    velocity_df = compute_velocity_from_id(
        record_id, bodypart=bodypart, individual=individual
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
) -> list[float]:
    """Compute one scalar speed summary per record id."""
    return [
        summarize_speed_from_id(
            record_id,
            bodypart=bodypart,
            how=how,
            likelihood_min=likelihood_min,
            individual=individual,
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