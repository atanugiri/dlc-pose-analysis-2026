from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

import scripts.db.db_utils as db_utils
import scripts.features.feature_summary as feature_summary


DEFAULT_BODYPARTS = ["Head", "Neck", "Midback", "Lowerback", "Tailbase"]
DEFAULT_SUMMARY_FEATURE = "head_body_misalignment"


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Convert 2D vectors to unit vectors, preserving NaNs for zero-length rows."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0, vectors / norm, np.nan)


def _angle_between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the signed angle between two sets of 2D vectors."""
    left_unit = _unit(left)
    right_unit = _unit(right)
    dot = np.sum(left_unit * right_unit, axis=1)
    cross_z = left_unit[:, 0] * right_unit[:, 1] - left_unit[:, 1] * right_unit[:, 0]
    return np.arctan2(cross_z, dot)


def _unwrap(values: np.ndarray) -> np.ndarray:
    """Unwrap angles to avoid discontinuities at +/- pi."""
    return np.unwrap(values)


def _load_bodyparts_xy(
    df: pd.DataFrame,
    *,
    bodyparts: list[str],
    individual: str | None = None,
    likelihood_threshold: float | None = 0.5,
) -> dict[str, np.ndarray]:
    """Load x/y coordinates for several bodyparts from a DLC dataframe."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC-style MultiIndex columns.")

    scorer = df.columns.get_level_values(0).unique().tolist()[0]
    scorer_df = df[scorer]

    if df.columns.nlevels >= 4:
        if individual is None:
            available = scorer_df.columns.get_level_values(0).unique().tolist()
            raise ValueError(
                "Multi-animal data requires `individual`. "
                f"Available individuals: {available}"
            )

        if individual not in scorer_df.columns.get_level_values(0):
            available = scorer_df.columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Individual {individual!r} not found. Available: {available}")

        target_df = scorer_df[individual]
    else:
        target_df = scorer_df

    out: dict[str, np.ndarray] = {}
    for bodypart in bodyparts:
        if bodypart not in target_df.columns.get_level_values(0):
            available = target_df.columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

        coords = target_df[bodypart]
        if not {"x", "y", "likelihood"}.issubset(coords.columns):
            raise ValueError(
                f"Bodypart {bodypart!r} is missing x/y/likelihood columns. Found: {list(coords.columns)}"
            )

        x = coords["x"].astype(float).to_numpy()
        y = coords["y"].astype(float).to_numpy()
        likelihood = coords["likelihood"].astype(float).to_numpy()

        if likelihood_threshold is not None:
            thr = float(likelihood_threshold)
            low_mask = likelihood < thr
            if low_mask.any():
                x = x.copy()
                y = y.copy()
                x[low_mask] = np.nan
                y[low_mask] = np.nan
                x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
                y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

        out[bodypart] = np.column_stack([x, y])

    return out


def compute_angle_features_from_df(
    df: pd.DataFrame,
    *,
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    bodyparts: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-frame angle features from a DLC dataframe.

    The default feature set uses Head, Neck, Midback, Lowerback, and Tailbase.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")

    selected_bodyparts = bodyparts or DEFAULT_BODYPARTS
    coords = _load_bodyparts_xy(
        df,
        bodyparts=selected_bodyparts,
        individual=individual,
        likelihood_threshold=likelihood_threshold,
    )

    head = coords["Head"]
    neck = coords["Neck"]
    midback = coords["Midback"]
    lowerback = coords["Lowerback"]
    tailbase = coords["Tailbase"]

    v_tail_head = head - tailbase
    v_head_neck = head - neck
    v_neck_mid = neck - midback
    v_mid_low = midback - lowerback
    v_low_tail = lowerback - tailbase

    head_body_misalignment = _angle_between(v_tail_head, v_head_neck)
    bend_neck = _angle_between(-v_head_neck, -v_neck_mid)
    bend_mid = _angle_between(-v_neck_mid, -v_mid_low)
    bend_low = _angle_between(-v_mid_low, -v_low_tail)

    if smoothing_window is not None:
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be a positive integer or None.")
        window = int(smoothing_window)
        if window > 1:
            if window % 2 == 0:
                window += 1
            window = max(3, window)
            head_body_misalignment = uniform_filter1d(
                _unwrap(head_body_misalignment.copy()),
                size=window,
                mode="nearest",
            )

    time = np.arange(len(head_body_misalignment), dtype=float) / float(fps)
    valid = (
        np.isfinite(head_body_misalignment)
        & np.isfinite(bend_neck)
        & np.isfinite(bend_mid)
        & np.isfinite(bend_low)
    )

    out = pd.DataFrame(
        {
            "time": time[valid],
            "head_body_misalignment": head_body_misalignment[valid],
            "bend_neck": bend_neck[valid],
            "bend_mid": bend_mid[valid],
            "bend_low": bend_low[valid],
        }
    )

    return out


def compute_angle_features_from_id(
    record_id: int,
    *,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    bodyparts: list[str] | None = None,
) -> pd.DataFrame:
    """Load one DB record and compute per-frame angle features."""
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    return compute_angle_features_from_df(
        df,
        fps=fps,
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
        bodyparts=bodyparts,
    )


def summarize_angle_features(
    angle_df: pd.DataFrame,
    *,
    how: str = "mean",
) -> float:
    """Summarize a scalar angle feature column from a per-frame feature DataFrame."""
    return feature_summary.summarize_feature(
        angle_df,
        feature_name=DEFAULT_SUMMARY_FEATURE,
        how=how,
    )


def summarize_angle_features_from_id(
    record_id: int,
    *,
    how: str = "mean",
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    bodyparts: list[str] | None = None,
) -> float:
    """Compute one scalar angle summary for one DB record ID."""
    angle_df = compute_angle_features_from_id(
        record_id,
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
        bodyparts=bodyparts,
    )

    return summarize_angle_features(angle_df, how=how)


def summarize_angle_features_from_ids(
    record_ids: list[int],
    *,
    how: str = "mean",
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    bodyparts: list[str] | None = None,
) -> list[float]:
    """Compute one scalar angle summary per record ID."""
    return [
        summarize_angle_features_from_id(
            record_id,
            how=how,
            individual=individual,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
            bodyparts=bodyparts,
        )
        for record_id in record_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute summarized angle values for one or more record IDs."
    )
    parser.add_argument(
        "record_ids",
        nargs="+",
        type=int,
        help="One or more database record IDs.",
    )
    parser.add_argument(
        "--how",
        default="mean",
        choices=["mean", "median", "max", "std"],
        help="Summary method.",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.5,
        help="Optional minimum likelihood threshold.",
    )
    parser.add_argument(
        "--individual",
        default=None,
        help="Optional individual name for multi-animal data.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Optional smoothing window size for head-body misalignment.",
    )

    args = parser.parse_args()

    values = summarize_angle_features_from_ids(
        args.record_ids,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
    )

    for record_id, value in zip(args.record_ids, values):
        print(f"{record_id}\t{value}")


if __name__ == "__main__":
    main()
