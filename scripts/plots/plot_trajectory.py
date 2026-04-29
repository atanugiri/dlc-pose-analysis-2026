from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
import scripts.utils.dlc_utils as dlc_utils


def plot_trajectory_from_df(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    color_by_time: bool = False,
):
    """Plot a single smooth trajectory from a DLC DataFrame."""
    x, y, _, time = dlc_utils.get_bodypart_xy_time(
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

        x, y = dlc_utils.get_bodypart_xy(
            df,
            bodypart=bodypart,
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