from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_pose_path(filtered_pose_file: str, *, repo_root: Path | None = None) -> Path:
    """Resolve a filtered pose path to an existing file on disk.

    The DB may store either an absolute/relative path or just a filename.
    If `repo_root` is not provided, uses `Path.cwd()`.
    """
    candidate = Path(filtered_pose_file)
    if candidate.exists():
        return candidate

    root = Path.cwd() if repo_root is None else Path(repo_root)
    for base in [root, root / "data" / "filtered_pose_data", root / "data" / "raw_pose_data"]:
        alt = base / filtered_pose_file
        if alt.exists():
            return alt

    raise FileNotFoundError(
        f"Could not find pose file on disk. Tried: {filtered_pose_file!r} and common data/ locations."
    )


def load_dlc_dataframe(h5_path: Path, *, preferred_key: str = "/df_with_missing") -> pd.DataFrame:
    """Load the main DeepLabCut DataFrame from an .h5 file."""
    try:
        return pd.read_hdf(h5_path, key=preferred_key)
    except (KeyError, ValueError):
        with pd.HDFStore(h5_path, mode="r") as store:
            keys = list(store.keys())
        if not keys:
            raise ValueError(f"No HDF keys found in {h5_path}")
        return pd.read_hdf(h5_path, key=keys[0])
