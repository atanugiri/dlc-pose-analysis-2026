from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.db.db_utils as db_utils


DEFAULT_BODYPARTS = ["Head", "Neck", "Tailbase"]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, v / n, np.nan)


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu, vv = _unit(u), _unit(v)
    dot = np.sum(uu * vv, axis=1)
    cross_z = uu[:, 0] * vv[:, 1] - uu[:, 1] * vv[:, 0]
    return np.arctan2(cross_z, dot)


def head_body_misalignment_from_df(
    df: pd.DataFrame,
    *,
    likelihood_threshold: float | None = 0.65,
    bodyparts: list[str] | None = None,
    individual: str | None = None,
) -> pd.Series:
    """Return per-frame absolute head-body misalignment as a pandas Series.

    This returns |theta| for each frame, aligned to the input DataFrame's index.
    """
    selected = bodyparts or DEFAULT_BODYPARTS

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC-style MultiIndex columns.")

    scorer = df.columns.get_level_values(0).unique().tolist()[0]

    # Multi-animal DLC files have 4+ column levels: (scorer, individual, bodypart, coord)
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

    def _read(bp: str) -> np.ndarray:
        if df.columns.nlevels >= 4:
            coords = df[scorer][individual][bp]
        else:
            coords = df[scorer][bp]

        if not {"x", "y"}.issubset(coords.columns):
            raise ValueError(f"Bodypart {bp!r} is missing x/y columns. Found: {list(coords.columns)}")

        x = coords["x"].astype(float).copy()
        y = coords["y"].astype(float).copy()

        if "likelihood" in coords.columns and likelihood_threshold is not None:
            p = coords["likelihood"].astype(float).copy()
            if likelihood_threshold is not None:
                x[p < likelihood_threshold] = np.nan
                y[p < likelihood_threshold] = np.nan

        xs = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
        ys = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)
        return np.column_stack([xs, ys])

    head = _read(selected[0])
    neck = _read(selected[1])
    tail = _read(selected[2])

    v_tail_head = head - tail
    v_head_neck = head - neck

    angles = _angle_between(v_tail_head, v_head_neck)
    abs_angles = np.abs(angles)

    return pd.Series(abs_angles, index=df.index, name="head_body_misalignment")


def head_body_misalignment_metrics(
    df: pd.DataFrame,
    *,
    likelihood_threshold: float | None = 0.5,
    bodyparts: list[str] | None = None,
    individual: str | None = None,
) -> dict:
    """Compute multiple head-body misalignment statistics.
    
    Returns a dict with p95, mean, median, and max values.
    """
    series = head_body_misalignment_from_df(
        df, likelihood_threshold=likelihood_threshold, bodyparts=bodyparts, individual=individual
    )
    arr = series.to_numpy()
    
    return {
        'p95': float(np.nanpercentile(arr, 95)),
        'mean': float(np.nanmean(arr)),
        'median': float(np.nanmedian(arr)),
        'max': float(np.nanmax(arr)),
    }


def head_body_misalignment_metrics_from_id(
    record_id: int,
    *,
    likelihood_threshold: float | None = 0.65,
    bodyparts: list[str] | None = None,
    individual: str | None = None,
) -> float:
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    # Use the new helper that returns the scalar p95
    return head_body_misalignment_metrics(
        df, likelihood_threshold=likelihood_threshold, bodyparts=bodyparts, individual=individual
    )

def head_body_misalignment_metrics_from_ids(
    record_ids: list[int],
    *,
    likelihood_threshold: float | None = 0.65,
    bodyparts: list[str] | None = None,
    individual: str | None = None,
) -> list[dict]:
    """Compute head-body misalignment metrics for a list of record IDs.

    Returns a list of dicts, each with keys: p95, mean, median, max.
    """
    return [
        head_body_misalignment_metrics_from_id(
            rid, likelihood_threshold=likelihood_threshold, bodyparts=bodyparts, individual=individual
        )
        for rid in record_ids
    ]

if __name__ == "__main__":
    record_ids = [1, 2]
    misalignment_p95_list = head_body_misalignment_metrics_from_ids(record_ids)
    for record_id, misalignment_p95 in zip(record_ids, misalignment_p95_list):
        print(f"Record ID {record_id} head-body misalignment p95: {misalignment_p95:.2f} radians")
