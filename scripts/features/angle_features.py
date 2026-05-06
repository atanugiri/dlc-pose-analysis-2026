from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import scripts.db.db_utils as db_utils
import scripts.features.feature_summary as feature_summary


DEFAULT_BODYPARTS = ["Head", "Neck", "Midback", "Lowerback", "Tailbase"]
DEFAULT_SUMMARY_FEATURE = "head_body_misalignment"


def _angle_of(v: np.ndarray) -> np.ndarray:
    """Compute the angle of 2D vectors."""
    return np.arctan2(v[:, 1], v[:, 0])


def _unit(v: np.ndarray) -> np.ndarray:
    """Convert vectors to unit vectors, handling zero-length vectors."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, v / n, np.nan)


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute the signed angle between two sets of 2D vectors."""
    uu, vv = _unit(u), _unit(v)
    dot = np.sum(uu * vv, axis=1)
    cross_z = uu[:, 0]*vv[:, 1] - uu[:, 1]*vv[:, 0]
    return np.arctan2(cross_z, dot)


def _unwrap(values: np.ndarray) -> np.ndarray:
    """Unwrap angles to avoid discontinuities at +/- pi."""
    return np.unwrap(values)


def compute_angle_features_from_df(
    df: pd.DataFrame,
    *,
    fps: float = db_utils.DEFAULT_FPS,
    individual: str | None = None,
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = 0.5,
    bodyparts: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-frame angle features from a DLC dataframe (OLD IMPLEMENTATION)."""
    if fps <= 0:
        raise ValueError("fps must be > 0")

    selected_bodyparts = bodyparts or DEFAULT_BODYPARTS
    
    # Handle multi-level columns (HDF5 format: scorer, bodypart, coord)
    # or 2-level columns (CSV format: bodypart, coord)
    if df.columns.nlevels == 3:
        # HDF5 format: extract scorer and access data
        scorer = df.columns.get_level_values(0).unique()[0]
        df = df[scorer]  # Drop scorer level
    
    # Load bodypart coordinates with likelihood filtering and interpolation
    B = {}
    for bp in selected_bodyparts:
        if bp not in df.columns.get_level_values(0):
            raise ValueError(f"Bodypart '{bp}' not found in dataframe")
        
        x = df[(bp, 'x')].astype(float).copy()
        y = df[(bp, 'y')].astype(float).copy()
        
        if (bp, 'likelihood') in df.columns:
            p = df[(bp, 'likelihood')].astype(float).copy()
            if likelihood_threshold is not None:
                x[p < likelihood_threshold] = np.nan
                y[p < likelihood_threshold] = np.nan
        
        # Interpolate NaN values
        x_interp = pd.Series(x).interpolate(limit_direction='both').to_numpy(dtype=float)
        y_interp = pd.Series(y).interpolate(limit_direction='both').to_numpy(dtype=float)
        B[bp] = np.column_stack([x_interp, y_interp])
    
    head = B["Head"]
    neck = B["Neck"]
    mid = B["Midback"]
    low = B["Lowerback"]
    tail = B["Tailbase"]

    v_tail_head = head - tail
    v_head_neck = head - neck
    v_neck_mid = neck - mid
    v_mid_low = mid - low
    v_low_tail = low - tail

    theta_body = _angle_of(v_tail_head)
    theta_head = _angle_of(v_head_neck)
    theta_seg_head_neck = _angle_of(v_head_neck)
    theta_seg_neck_mid = _angle_of(v_neck_mid)
    theta_seg_mid_low = _angle_of(v_mid_low)
    theta_seg_low_tail = _angle_of(v_low_tail)

    head_body_misalignment = _angle_between(v_tail_head, v_head_neck)
    bend_neck = _angle_between(-v_head_neck, -v_neck_mid)
    bend_mid = _angle_between(-v_neck_mid, -v_mid_low)
    bend_low = _angle_between(-v_mid_low, -v_low_tail)
    tail_bend_index = np.abs(bend_neck) + np.abs(bend_mid) + np.abs(bend_low)

    theta_body_u = _unwrap(theta_body.copy())
    if smoothing_window and smoothing_window >= 3:
        if smoothing_window % 2 == 0:
            smoothing_window += 1
        k = smoothing_window // 2
        pad = np.pad(theta_body_u, (k, k), mode='edge')
        kern = np.ones(smoothing_window) / smoothing_window
        theta_body_u = np.convolve(pad, kern, mode='valid')

    dt = 1.0 / float(fps)
    ang_vel_body = np.gradient(theta_body_u, dt)   # rad/s

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
            "tail_bend_index": tail_bend_index[valid],
            "ang_vel_body": ang_vel_body[valid],
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


def batch_angle_features(
    dlc_table: pd.DataFrame,
    record_ids: list[int],
    likelihood_threshold: float = 0.5,
    smooth_window: int | None = None,
) -> pd.DataFrame:
    """
    Compute full summary statistics for multiple trials.
    
    Returns a DataFrame with columns:
    - trial_id, trial_length_s, minutes, frame_rate (metadata)
    - head_body_misalignment_mean, head_body_misalignment_p95
    - tail_bend_index_mean, tail_bend_index_p95
    - abs_bend_neck_mean, abs_bend_mid_mean, abs_bend_low_mean
    - ang_vel_body_mean_s, ang_vel_body_p95_s
    - ang_vel_body_mean_min, ang_vel_body_p95_min
    
    Args:
        dlc_table: Ignored (for API compatibility with old code)
        record_ids: List of trial IDs to process
        likelihood_threshold: Minimum likelihood to trust a point
        smooth_window: Window size for body angle smoothing (optional)
    
    Returns:
        DataFrame with one row per trial, all summary statistics as columns
    """
    rows = []
    
    for trial_id in record_ids:
        try:
            # Get trial metadata
            trial_length_s = None
            try:
                conn = db_utils.connect()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT trial_length_s FROM public.experimental_metadata WHERE id = %s",
                            (trial_id,),
                        )
                        row = cur.fetchone()
                        if row:
                            trial_length_s = float(row[0])
                finally:
                    conn.close()
            except Exception:
                pass
            
            fps = db_utils.get_fps(trial_id)
            minutes = trial_length_s / 60.0 if trial_length_s and trial_length_s > 0 else np.nan
            
            # Compute per-frame angle features
            angle_df = compute_angle_features_from_id(
                trial_id,
                individual=None,
                smoothing_window=smooth_window,
                likelihood_threshold=likelihood_threshold,
                bodyparts=None,
            )
            
            # Compute statistics on each feature
            def _stats(x):
                x = x[np.isfinite(x)]
                if x.size == 0:
                    return dict(mean=np.nan, p95=np.nan)
                return dict(
                    mean=float(np.nanmean(x)),
                    p95=float(np.nanpercentile(x, 95)),
                )
            
            # Per-minute scaling for angular velocity
            ang_vel_mean = float(np.nanmean(angle_df['ang_vel_body'].to_numpy()))
            ang_vel_p95 = float(np.nanpercentile(angle_df['ang_vel_body'].to_numpy(), 95))
            
            rows.append(dict(
                trial_id=trial_id,
                trial_length_s=trial_length_s,
                minutes=minutes,
                frame_rate=fps,
                
                # Orientation/bend magnitudes (radians)
                head_body_misalignment_mean=_stats(angle_df['head_body_misalignment'].to_numpy())['mean'],
                head_body_misalignment_p95=_stats(angle_df['head_body_misalignment'].to_numpy())['p95'],
                tail_bend_index_mean=_stats(angle_df['tail_bend_index'].to_numpy())['mean'],
                tail_bend_index_p95=_stats(angle_df['tail_bend_index'].to_numpy())['p95'],
                
                # Absolute bend magnitudes
                abs_bend_neck_mean=_stats(np.abs(angle_df['bend_neck'].to_numpy()))['mean'],
                abs_bend_mid_mean=_stats(np.abs(angle_df['bend_mid'].to_numpy()))['mean'],
                abs_bend_low_mean=_stats(np.abs(angle_df['bend_low'].to_numpy()))['mean'],
                
                # Angular velocity (rad/s and rad/min)
                ang_vel_body_mean_s=ang_vel_mean,
                ang_vel_body_p95_s=ang_vel_p95,
                ang_vel_body_mean_min=ang_vel_mean * 60.0 if np.isfinite(ang_vel_mean) else np.nan,
                ang_vel_body_p95_min=ang_vel_p95 * 60.0 if np.isfinite(ang_vel_p95) else np.nan,
            ))
            
        except Exception as e:
            print(f"[WARN] Failed to process trial {trial_id}: {e}")
            continue
    
    return pd.DataFrame(rows)


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
