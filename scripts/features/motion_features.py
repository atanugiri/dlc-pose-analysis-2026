from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.features.feature_summary as feature_summary
from scripts.features.normalize_pose import (
    normalize_bodypart_from_id,
    get_bodypart_from_id,
)


def compute_velocity_from_id(
    record_id: int,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.9,
    normalization: bool = True,
) -> pd.DataFrame:
    """Load one DB record and compute per-frame velocity.

    When normalization=True, coordinates are normalized to unit square using
    pooled maze corners. When False, raw DLC coordinates are used.
    """
    if normalization:
        x, y, likelihood, time, index = normalize_bodypart_from_id(
            record_id,
            bodypart=bodypart,
            individual=individual,
            likelihood_threshold=likelihood_threshold,
            smoothing_window=smoothing_window,
        )
    else:
        x, y, likelihood, time, index = get_bodypart_from_id(
            record_id,
            bodypart=bodypart,
            individual=individual,
            likelihood_threshold=likelihood_threshold,
            smoothing_window=smoothing_window,
        )

    # Convert arrays to Series with proper index
    x = pd.Series(x, index=index, dtype=float)
    y = pd.Series(y, index=index, dtype=float)
    likelihood = pd.Series(likelihood, index=index, dtype=float)
    t = pd.Series(time, index=index).astype(float)

    # Compute velocities
    dt = t.diff().replace(0, np.nan)
    vx = x.diff() / dt
    vy = y.diff() / dt
    speed = np.hypot(vx, vy)

    return pd.DataFrame({
        "time": t,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "likelihood": likelihood,
    })


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
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.9,
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
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.9,
    normalization: bool = True,
) -> list[float]:
    """Compute one scalar speed summary per record id."""
    return [
        summarize_speed_from_id(
            record_id,
            bodypart=bodypart,
            how=how,
            individual=individual,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
            normalization=normalization,
        )
        for record_id in record_ids
    ]



if __name__ == "__main__":
    record_ids = [1, 2]
    values = summarize_speed_from_ids(record_ids, bodypart="Midback", how="mean", likelihood_threshold=0.9, normalization=True)
    print(values)
