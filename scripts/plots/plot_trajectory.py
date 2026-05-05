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
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    frame_height: int | None = None,
):
    """Plot a single smooth trajectory from a DLC DataFrame."""
    x, y, likelihood, time, _ = dlc_utils.get_bodypart_xy_time(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
    )

    # Invert y-axis if frame_height is provided (DeepLabCut inverts y-scale)
    if frame_height is not None:
        y = frame_height - y

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
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
):
    """Plot trajectory of a bodypart from a record ID."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)
    _, frame_height = db_utils.get_frame_dimensions(record_id)

    plot_trajectory_from_df(
        df,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        color_by_time=color_by_time,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
        frame_height=frame_height,
    )

def plot_trajectory_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    linewidth: float = 1.0,
    alpha: float = 0.7,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
):
    """Overlay smooth trajectories of a bodypart from multiple record IDs."""
    plt.figure(figsize=(6, 6))

    for record_id in record_ids:
        filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
        df = db_utils.load_dlc_dataframe(filtered_pose_file)
        fps = db_utils.get_fps(record_id)
        _, frame_height = db_utils.get_frame_dimensions(record_id)

        x, y, _, _, _ = dlc_utils.get_bodypart_xy_time(
            df,
            bodypart=bodypart,
            fps=fps,
            individual=individual,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
        )

        # Invert y-axis (DeepLabCut inverts y-scale)
        y = frame_height - y

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
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Optional smoothing window size for trajectory smoothing.",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.5,
        help="Likelihood threshold for filtering low-confidence poses.",
    )
    args = parser.parse_args()

    plot_trajectory_from_ids(
        args.ids,
        bodypart=args.bodypart,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
    )