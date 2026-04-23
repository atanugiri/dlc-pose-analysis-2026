"""
Keypoint-MoSeq (KPMS) Syllable Pipeline
========================================
Loads filtered DLC CSV pose data, fits a KPMS model, and produces a
frame-by-frame syllable time-series CSV for each recording.

Usage
-----
    python scripts/kpms_pipeline.py [--project-dir kpms_project] \
                                    [--pose-dir filtered_pose_data] \
                                    [--num-iters 200] [--kappa 1e4]

The script expects filtered DLC CSV files (multi-index header, scorer row +
bodypart row + coord row) in --pose-dir relative to the DLC project root
defined by DLC_PROJECT_ROOT (defaults to the directory two levels above this
repo, i.e. ../dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04).

Outputs
-------
results/
    syllable_timeseries.csv   – one row per frame, columns: recording, frame,
                                syllable, time_sec (at 30 fps by default)
kpms_project/
    config.yml                – KPMS configuration
    pca.p                     – fitted PCA
    <model_name>/             – saved model checkpoints
    results.h5                – raw KPMS results
    results.csv               – alias for syllable_timeseries.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run keypoint-moseq on DLC CSV data")
    p.add_argument(
        "--project-dir",
        default="kpms_project",
        help="Directory where KPMS config/model will be saved (default: kpms_project)",
    )
    p.add_argument(
        "--pose-dir",
        default=None,
        help=(
            "Path to directory containing filtered DLC CSV files. "
            "Defaults to <DLC_PROJECT_ROOT>/filtered_pose_data"
        ),
    )
    p.add_argument(
        "--raw-pose-dir",
        default=None,
        help=(
            "Path to directory containing raw DLC CSV files. "
            "Used only when --pose-dir is not supplied and filtered data are absent. "
            "Defaults to <DLC_PROJECT_ROOT>/raw_pose_data"
        ),
    )
    p.add_argument(
        "--video-dir",
        default=None,
        help=(
            "Path to directory containing video files (used for syllable clip export). "
            "Defaults to <DLC_PROJECT_ROOT>/videos"
        ),
    )
    p.add_argument(
        "--dlc-config",
        default=None,
        help="Path to DLC config.yaml (used to read bodypart list). Optional.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second of the recordings (default: 30)",
    )
    p.add_argument(
        "--num-ar-iters",
        type=int,
        default=50,
        help="Number of AR-only fitting iterations (default: 50)",
    )
    p.add_argument(
        "--num-iters",
        type=int,
        default=200,
        help="Number of full model fitting iterations (default: 200)",
    )
    p.add_argument(
        "--kappa",
        type=float,
        default=None,
        help=(
            "Sticky parameter kappa controlling syllable duration. "
            "Leave unset to use KPMS auto-calibration."
        ),
    )
    p.add_argument(
        "--num-syllables",
        type=int,
        default=100,
        help="Truncation level (max syllables) for the HDP prior (default: 100)",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="DLC confidence threshold below which keypoints are masked (default: 0.5)",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory for output CSVs (default: results)",
    )
    p.add_argument(
        "--export-clips",
        action="store_true",
        default=False,
        help="Export short video clips for each syllable (requires --video-dir)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: resolve DLC project root
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DLC_ROOT = os.path.join(
    os.path.dirname(REPO_ROOT),
    "dlc-pose-estimation",
    "ElevatedMazeFood-Atanu-2026-04-04",
)


def _resolve(path_arg, default_subdir):
    if path_arg:
        return path_arg
    candidate = os.path.join(DEFAULT_DLC_ROOT, default_subdir)
    return candidate


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        import keypoint_moseq as kpms
    except ImportError:
        sys.exit(
            "keypoint-moseq is not installed.\n"
            "Install it with:  conda env create -f environment.yml\n"
            "or:               pip install keypoint-moseq"
        )

    pose_dir = _resolve(args.pose_dir, "filtered_pose_data")
    raw_pose_dir = _resolve(args.raw_pose_dir, "raw_pose_data")
    video_dir = _resolve(args.video_dir, "videos")
    project_dir = args.project_dir
    results_dir = args.results_dir

    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Locate CSV files
    # ------------------------------------------------------------------
    csv_dir = pose_dir if os.path.isdir(pose_dir) else raw_pose_dir
    if not os.path.isdir(csv_dir):
        sys.exit(
            f"Pose data directory not found: {csv_dir}\n"
            "Pass --pose-dir or --raw-pose-dir explicitly."
        )

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        sys.exit(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {csv_dir}")

    # ------------------------------------------------------------------
    # 2. Setup KPMS project (writes config.yml if absent)
    # ------------------------------------------------------------------
    config_path = os.path.join(project_dir, "config.yml")
    if not os.path.exists(config_path):
        if args.dlc_config and os.path.exists(args.dlc_config):
            kpms.setup_project(project_dir, deeplabcut_config=args.dlc_config)
        else:
            # Infer bodyparts from first CSV file
            sample_csv = os.path.join(csv_dir, csv_files[0])
            bodyparts = _read_bodyparts_from_csv(sample_csv)
            print(f"Detected bodyparts: {bodyparts}")
            kpms.setup_project(project_dir, bodyparts=bodyparts)

    config = kpms.load_config(project_dir)

    # Apply CLI overrides to config
    if args.kappa is not None:
        config["kappa"] = args.kappa
    config["num_states"] = args.num_syllables
    config["conf_pseudocount"] = args.conf_threshold

    # ------------------------------------------------------------------
    # 3. Load DLC CSV data
    # ------------------------------------------------------------------
    print("Loading DLC results …")
    coordinates, confidences = kpms.load_deeplabcut_results(
        csv_dir,
        # Use filtered files (CSV) by passing the directory
    )

    print(f"Loaded {len(coordinates)} recording(s)")

    # ------------------------------------------------------------------
    # 4. Fit PCA
    # ------------------------------------------------------------------
    pca_path = os.path.join(project_dir, "pca.p")
    if os.path.exists(pca_path):
        print("Loading existing PCA …")
        pca = kpms.load_pca(project_dir)
    else:
        print("Fitting PCA …")
        pca = kpms.fit_pca(**kpms.format_data(coordinates, confidences, **config), **config)
        kpms.save_pca(pca, project_dir)
        kpms.print_dims_to_explain_variance(pca, 0.90)

    # ------------------------------------------------------------------
    # 5. Format data for model fitting
    # ------------------------------------------------------------------
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    # ------------------------------------------------------------------
    # 6. Fit AR-HMM (AR-only pass, then full pass)
    # ------------------------------------------------------------------
    model = kpms.init_model(data, pca, **config)

    print(f"AR-only fitting ({args.num_ar_iters} iterations) …")
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        ar_only=True,
        num_iters=args.num_ar_iters,
    )

    print(f"Full model fitting ({args.num_iters} iterations) …")
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        num_iters=args.num_iters,
        model_name=model_name,
    )

    # ------------------------------------------------------------------
    # 7. Apply model and extract syllable sequences
    # ------------------------------------------------------------------
    print("Applying model to extract syllables …")
    results = kpms.apply_model(
        model,
        pca,
        data,
        metadata,
        project_dir,
        **config,
    )

    # Save raw KPMS results (h5 + per-recording CSVs)
    kpms.save_results_as_csv(results, project_dir)

    # ------------------------------------------------------------------
    # 8. Build unified syllable time-series CSV
    # ------------------------------------------------------------------
    rows = []
    for recording, rec_results in results.items():
        syllables = np.array(rec_results["syllable"])
        n_frames = len(syllables)
        times = np.arange(n_frames) / args.fps
        for frame_idx, (syl, t) in enumerate(zip(syllables, times)):
            rows.append(
                {
                    "recording": recording,
                    "frame": frame_idx,
                    "syllable": int(syl),
                    "time_sec": round(t, 4),
                }
            )

    df = pd.DataFrame(rows, columns=["recording", "frame", "syllable", "time_sec"])
    out_csv = os.path.join(results_dir, "syllable_timeseries.csv")
    df.to_csv(out_csv, index=False)
    print(f"Syllable time-series saved to {out_csv}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # 9. (Optional) Export syllable video clips
    # ------------------------------------------------------------------
    if args.export_clips:
        if not os.path.isdir(video_dir):
            print(f"Warning: --export-clips requested but video directory not found: {video_dir}")
        else:
            clips_dir = os.path.join(results_dir, "syllable_clips")
            os.makedirs(clips_dir, exist_ok=True)
            print(f"Exporting syllable clips to {clips_dir} …")
            kpms.generate_grid_movies(
                results,
                project_dir,
                video_dir=video_dir,
                output_dir=clips_dir,
            )

    print("Done.")


# ---------------------------------------------------------------------------
# Helper: read bodyparts from DLC multi-index CSV
# ---------------------------------------------------------------------------

def _read_bodyparts_from_csv(csv_path):
    """
    DLC CSVs have three header rows:
        row 0: scorer  (repeated scorer name)
        row 1: bodyparts
        row 2: coords  (x, y, likelihood)
    Returns deduplicated list of bodyparts (preserving order).
    """
    header = pd.read_csv(csv_path, header=None, nrows=3)
    bodypart_row = header.iloc[1, 1:].tolist()  # skip first 'bodyparts' label
    seen = set()
    bodyparts = []
    for bp in bodypart_row:
        if bp not in seen:
            seen.add(bp)
            bodyparts.append(bp)
    return bodyparts


if __name__ == "__main__":
    main()
