from pathlib import Path
import pandas as pd


def load_dlc_dataframe(filtered_pose_file, repo_root="."):
    h5_path = Path(repo_root) / "data" / "filtered_pose_data" / filtered_pose_file

    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    return pd.read_hdf(h5_path, key="/df_with_missing")
