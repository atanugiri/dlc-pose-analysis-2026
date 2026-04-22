from pathlib import Path
import pandas as pd


def load_dlc_dataframe(filtered_pose_file, repo_root="."):
    h5_path = Path(repo_root) / "data" / "filtered_pose_data" / filtered_pose_file

    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    return pd.read_hdf(h5_path, key="/df_with_missing")


if __name__ == "__main__":
    import argparse
    import db_utils

    parser = argparse.ArgumentParser()
    parser.add_argument("record_id", type=int)
    args = parser.parse_args()

    filtered_pose_file = db_utils.get_filtered_pose_file(args.record_id)

    df = load_dlc_dataframe(filtered_pose_file)
    print(df.head())