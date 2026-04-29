from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import scripts.db.db_utils as db_utils
import scripts.features.feature_summary as feature_summary


def compute_velocity_from_df(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
) -> pd.DataFrame:
    """Compute per-frame x, y, vx, vy, and speed for one DLC bodypart."""
    if fps <= 0:
        raise ValueError("fps must be > 0")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(
            "Expected DLC-style MultiIndex columns: scorer, bodypart, coord."
        )

    # Handle multi-animal files with 4-level columns: (scorer, individual, bodypart, coord)
    # If present, require `individual` to be passed and convert to the 3-level
    # format (scorer, bodypart, coord) expected by the rest of the function.
    if df.columns.nlevels >= 4:
        scorer = df.columns.get_level_values(0).unique().tolist()[0]
        if individual is None:
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(
                "DataFrame contains an individual level; pass `individual` (e.g. 'm1')"
                f" to select one. Available individuals: {available}"
            )

        if individual not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Individual {individual!r} not found. Available: {available}")

        coords = df[scorer][individual]

        # coords currently has columns like (bodypart, coord). Rebuild a 3-level
        # MultiIndex with scorer as the top level so downstream logic can remain.
        tuples = [(scorer, bp, coord) for bp, coord in coords.columns]
        new_cols = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodypart", "coord"])
        coords.columns = new_cols
        df = coords

    if df.columns.nlevels < 3:
        raise ValueError(
            "Expected DLC-style MultiIndex columns: scorer, bodypart, coord."
        )

    scorer = df.columns.get_level_values(0).unique().tolist()[0]

    if bodypart not in df[scorer].columns.get_level_values(0):
        available = df[scorer].columns.get_level_values(0).unique().tolist()
        raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

    coords = df[scorer][bodypart]

    if not {"x", "y"}.issubset(coords.columns):
        raise ValueError(
            f"Bodypart {bodypart!r} is missing x/y columns. Found: {list(coords.columns)}"
        )

    x = coords["x"].astype(float)
    y = coords["y"].astype(float)

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

    if "likelihood" in coords.columns:
        out["likelihood"] = coords["likelihood"].astype(float)

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