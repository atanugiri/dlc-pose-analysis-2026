from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
import scripts.features.feature_summary as feature_summary

def _get_trajectory_xy_time(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
):
    """Extract x, y, and time arrays from a DLC-style DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC-style MultiIndex columns.")

    scorer = df.columns.get_level_values(0).unique().tolist()[0]

    if df.columns.nlevels >= 4:
        if individual is None:
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(
                "Multi-animal data requires `individual`. "
                f"Available individuals: {available}"
            )

        if individual not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Individual {individual!r} not found. Available: {available}")

        coords = df[scorer][individual][bodypart]

    else:
        if bodypart not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

        coords = df[scorer][bodypart]

    if not {"x", "y"}.issubset(coords.columns):
        raise ValueError(
            f"Bodypart {bodypart!r} is missing x/y columns. Found: {list(coords.columns)}"
        )

    x = coords["x"].astype(float).to_numpy()
    y = coords["y"].astype(float).to_numpy()
    time = np.arange(len(x)) / fps

    valid = np.isfinite(x) & np.isfinite(y)

    return x[valid], y[valid], time[valid]

def plot_trajectory_from_df(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    color_by_time: bool = False,
):
    """Plot a single smooth trajectory from a DLC DataFrame."""
    x, y, time = _get_trajectory_xy_time(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
    )

    plt.figure(figsize=(6, 6))

    if color_by_time:
        sc = plt.scatter(x, y, c=time, s=4)
        plt.colorbar(sc, label="Time (s)")
    else:
        plt.plot(x, y, linewidth=1.5)

    plt.title(f"Trajectory of {bodypart}")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)
    plt.show()

def plot_trajectory_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    color_by_time: bool = False,
):
    """Plot trajectory of a bodypart from a record ID."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    plot_trajectory_from_df(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        color_by_time=color_by_time,
    )

def plot_trajectory_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    linewidth: float = 1.0,
    alpha: float = 0.7,
):
    """Overlay smooth trajectories of a bodypart from multiple record IDs."""
    plt.figure(figsize=(6, 6))

    for record_id in record_ids:
        filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
        df = db_utils.load_dlc_dataframe(filtered_pose_file)
        fps = db_utils.get_fps(record_id)

        x, y, time = _get_trajectory_xy_time(
            df,
            bodypart=bodypart,
            fps=fps,
            individual=individual,
        )

        plt.plot(
            x,
            y,
            linewidth=linewidth,
            alpha=alpha,
            label=f"ID {record_id}",
        )

    plt.title(f"{bodypart} trajectories")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.3)

    if len(record_ids) <= 10:
        plt.legend(frameon=False)

    plt.show()

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Plot trajectories from DB record IDs.")
        parser.add_argument("ids", nargs="+", type=int, help="One or more record IDs to plot")
        parser.add_argument("--bodypart", default="Midback", help="Bodypart to plot")
        parser.add_argument("--individual", default=None, help="Individual (for multi-animal data)")
        args = parser.parse_args()

        plot_trajectory_from_ids(
            args.ids,
            bodypart=args.bodypart,
            individual=args.individual,
        )