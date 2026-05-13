from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
import scripts.utils.dlc_utils as dlc_utils
from scripts.features.normalize_pose import (
    normalize_bodypart_from_id,
    get_bodypart_from_id,
)


TRAJECTORY_LINEWIDTH = 1.0
TRAJECTORY_ALPHA = 0.7


def plot_trajectory_from_df(
    df: pd.DataFrame | None = None,
    *,
    bodypart: str = "Midback",
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    color_by_time: bool = False,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    frame_width: int | None = None,
    frame_height: int | None = None,
    ax=None,
    show: bool = True,
    label: str | None = None,
    x=None,
    y=None,
    likelihood=None,
    time=None,
):
    """Plot a single smooth trajectory from a DLC DataFrame or pre-computed coordinates.
    
    If x, y, likelihood, time are provided, they are used directly.
    Otherwise, they are extracted from the DataFrame.
    """
    if x is None or y is None or likelihood is None or time is None:
        if df is None:
            raise ValueError("Either df or (x, y, likelihood, time) must be provided")
        x, y, likelihood, time, _ = dlc_utils.get_bodypart_xy_time(
            df,
            bodypart=bodypart,
            fps=fps,
            individual=individual,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
        )

    if frame_height is not None:
        y = frame_height - y

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if color_by_time:
        sc = ax.scatter(x, y, c=time, s=4, alpha=TRAJECTORY_ALPHA, label=label)
        plt.colorbar(sc, ax=ax, label="Time (s)")
    else:
        ax.plot(x, y, linewidth=TRAJECTORY_LINEWIDTH, alpha=TRAJECTORY_ALPHA, label=label)

    ax.set_title(f"Trajectory of {bodypart}")
    ax.set_aspect("equal", adjustable="box")

    if frame_width is not None and frame_height is not None:
        ax.set_xlim(0, frame_width)
        ax.set_ylim(0, frame_height)

    if show:
        plt.show()

    return ax

def plot_trajectory_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    color_by_time: bool = False,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    show: bool = True,
    normalization: bool = True,
):
    """Plot trajectory of a bodypart from a record ID.
    
    When normalization=True, coordinates are normalized to unit square using
    pooled maze corners. When False, raw DLC coordinates are used.
    """
    fps = db_utils.get_fps(record_id)
    
    if normalization:
        x, y, likelihood, time, _ = normalize_bodypart_from_id(
            record_id,
            bodypart=bodypart,
            individual=individual,
            likelihood_threshold=likelihood_threshold,
            smoothing_window=smoothing_window,
        )
        frame_width, frame_height = 1, 1
    else:
        x, y, likelihood, time, _ = get_bodypart_from_id(
            record_id,
            bodypart=bodypart,
            individual=individual,
            likelihood_threshold=likelihood_threshold,
            smoothing_window=smoothing_window,
        )
        frame_width, frame_height = db_utils.get_frame_dimensions(record_id)

    return plot_trajectory_from_df(
        df=None,
        bodypart=bodypart,
        fps=fps,
        individual=individual,
        color_by_time=color_by_time,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
        frame_width=frame_width,
        frame_height=frame_height,
        label=f"ID {record_id}",
        show=show,
        x=x,
        y=y,
        likelihood=likelihood,
        time=time,
    )

def plot_trajectory_from_ids(
    record_ids: list[int],
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    show: bool = True,
    normalization: bool = True,
):
    """Overlay smooth trajectories of a bodypart from multiple record IDs.
    
    When normalization=True, coordinates are normalized to unit square using
    pooled maze corners. When False, raw DLC coordinates are used.
    """
    if not record_ids:
        raise ValueError("record_ids must contain at least one ID.")

    fig, ax = plt.subplots(figsize=(6, 6))

    if normalization:
        frame_width, frame_height = 1, 1
    else:
        frame_width, frame_height = db_utils.get_frame_dimensions(record_ids[0])

    for record_id in record_ids:
        fps = db_utils.get_fps(record_id)
        
        if normalization:
            x, y, likelihood, time, _ = normalize_bodypart_from_id(
                record_id,
                bodypart=bodypart,
                individual=individual,
                likelihood_threshold=likelihood_threshold,
                smoothing_window=smoothing_window,
            )
        else:
            x, y, likelihood, time, _ = get_bodypart_from_id(
                record_id,
                bodypart=bodypart,
                individual=individual,
                likelihood_threshold=likelihood_threshold,
                smoothing_window=smoothing_window,
            )
            width, height = db_utils.get_frame_dimensions(record_id)
            if width != frame_width or height != frame_height:
                print(
                    f"Warning: Record {record_id} has different dimensions "
                    f"({width}x{height}) than first record "
                    f"({frame_width}x{frame_height})"
                )

        plot_trajectory_from_df(
            df=None,
            bodypart=bodypart,
            fps=fps,
            individual=individual,
            color_by_time=False,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
            frame_width=frame_width,
            frame_height=frame_height,
            ax=ax,
            show=False,
            label=f"ID {record_id}",
            x=x,
            y=y,
            likelihood=likelihood,
            time=time,
        )

    ax.set_title(f"{bodypart} trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)

    if len(record_ids) <= 10:
        ax.legend(frameon=False)

    if show:
        plt.show()

    return ax