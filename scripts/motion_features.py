#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _resolve_pose_path(filtered_pose_file: str) -> Path:
    """Resolve the filtered pose path from DB to an existing file on disk."""
    candidate = Path(filtered_pose_file)
    if candidate.exists():
        return candidate

    # Common case: DB stores only the filename.
    repo_root = Path(__file__).resolve().parent
    repo_root = repo_root.parent if repo_root.name == "scripts" else repo_root
    for base in [repo_root, repo_root / "data" / "filtered_pose_data", repo_root / "data" / "raw_pose_data"]:
        alt = base / filtered_pose_file
        if alt.exists():
            return alt

    raise FileNotFoundError(
        f"Could not find pose file on disk. Tried: {filtered_pose_file!r} and common data/ locations."
    )


def _load_dlc_dataframe(h5_path: Path) -> pd.DataFrame:
    """Load the main DeepLabCut DataFrame from an .h5 file."""
    preferred_key = "/df_with_missing"
    try:
        return pd.read_hdf(h5_path, key=preferred_key)
    except (KeyError, ValueError):
        with pd.HDFStore(h5_path, mode="r") as store:
            keys = list(store.keys())
        if not keys:
            raise ValueError(f"No HDF keys found in {h5_path}")
        return pd.read_hdf(h5_path, key=keys[0])


def compute_velocity_from_h5(
    h5_path: Path,
    bodypart: str = "Midback",
    fps: float = 15.0,
) -> pd.DataFrame:
    """Compute per-frame (vx, vy, speed) for a given bodypart from a DLC .h5."""
    if fps <= 0:
        raise ValueError("fps must be > 0")

    df = _load_dlc_dataframe(h5_path)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
        raise ValueError(
            "Expected DLC-style MultiIndex columns (scorer, bodypart, coord). "
            f"Got: {type(df.columns).__name__}"
        )

    scorer = df.columns.get_level_values(0).unique().tolist()[0]
    if bodypart not in df[scorer].columns.get_level_values(0):
        available = df[scorer].columns.get_level_values(0).unique().tolist()
        raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

    coords = df[scorer][bodypart]
    if not {"x", "y"}.issubset(set(coords.columns)):
        raise ValueError(f"Bodypart {bodypart!r} is missing x/y columns. Found: {list(coords.columns)}")

    x = coords["x"].astype(float)
    y = coords["y"].astype(float)

    dt = 1.0 / float(fps)
    vx = x.diff() / dt
    vy = y.diff() / dt
    speed = np.sqrt(vx * vx + vy * vy)

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


def get_velocity(record_id: int, bodypart: str = "Midback", fps: float = 15.0) -> None:
    """Fetch the filtered pose file for a DB record id and print velocity stats."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="atanugiri",
            password="",
            database="dlc_pose_analysis_2026",
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filtered_pose_file FROM public.experimental_metadata WHERE id = %s",
            (record_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
    except ImportError as exc:
        raise SystemExit(
            "psycopg2 is not installed in this Python environment. "
            "Install it (e.g. `conda install -n ghrelin -c conda-forge psycopg2`) "
            "or use --h5-path to compute velocity directly from a file.\n"
            f"Import error: {exc}"
        )
    except Exception as exc:
        raise SystemExit(f"Database error: {exc}")

    if not row or not row[0]:
        raise SystemExit(f"No filtered_pose_file found for ID: {record_id}")

    pose_path = _resolve_pose_path(row[0])
    vel = compute_velocity_from_h5(pose_path, bodypart=bodypart, fps=fps)

    print(f"ID: {record_id}")
    print(f"Pose file: {pose_path}")
    print(f"Bodypart: {bodypart}")
    print(f"FPS: {fps}")

    speed = vel["speed"].replace([np.inf, -np.inf], np.nan)
    print(f"Frames: {len(vel)}")
    print(f"Mean speed (px/s): {float(speed.mean(skipna=True)):.3f}")
    print(f"Median speed (px/s): {float(speed.median(skipna=True)):.3f}")

    print("\nPreview:")
    with pd.option_context("display.max_columns", 10, "display.width", 140):
        print(vel.head(5))


def print_velocity_summary(h5_path: Path, bodypart: str, fps: float) -> None:
    vel = compute_velocity_from_h5(h5_path, bodypart=bodypart, fps=fps)

    print(f"Pose file: {h5_path}")
    print(f"Bodypart: {bodypart}")
    print(f"FPS: {fps}")

    speed = vel["speed"].replace([np.inf, -np.inf], np.nan)
    print(f"Frames: {len(vel)}")
    print(f"Mean speed (px/s): {float(speed.mean(skipna=True)):.3f}")
    print(f"Median speed (px/s): {float(speed.median(skipna=True)):.3f}")

    print("\nPreview:")
    with pd.option_context("display.max_columns", 10, "display.width", 140):
        print(vel.head(5))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame velocity from a DeepLabCut filtered .h5. "
            "You can either provide a DB record id (to look up filtered_pose_file) or pass --h5-path directly."
        )
    )
    parser.add_argument("id", type=int, nargs="?", help="experimental_metadata.id (optional if --h5-path provided)")
    parser.add_argument("--h5-path", type=Path, default=None, help="Path to a DLC .h5 (bypasses the database)")
    parser.add_argument("--bodypart", default="Midback", help="Bodypart name (default: Midback)")
    parser.add_argument("--fps", type=float, default=15.0, help="Frames per second (default: 15)")
    args = parser.parse_args()

    if args.h5_path is not None:
        if not args.h5_path.exists():
            raise FileNotFoundError(f"File not found: {args.h5_path}")
        print_velocity_summary(args.h5_path, bodypart=args.bodypart, fps=args.fps)
        return

    if args.id is None:
        raise ValueError("Provide either an experimental_metadata id or --h5-path")

    get_velocity(args.id, bodypart=args.bodypart, fps=args.fps)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
