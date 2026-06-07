from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_h5(csv_path: Path, overwrite: bool = False) -> Path:
    h5_path = csv_path.with_suffix(".h5")
    if h5_path.exists() and not overwrite:
        return h5_path

    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    df.to_hdf(h5_path, key="df_with_missing", mode="w", complevel=5, complib="blosc")
    return h5_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DLC filtered CSV files to HDF5 format."
    )
    parser.add_argument("directories", nargs="+", type=Path, help="Directories to search for CSV files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing H5 files.")
    args = parser.parse_args()

    csvs: list[Path] = []
    for directory in args.directories:
        csvs.extend(sorted(directory.glob("*.csv")))

    if not csvs:
        print("No CSV files found.")
        return

    print(f"Converting {len(csvs)} CSV files...")
    for csv_path in csvs:
        h5_path = convert_csv_to_h5(csv_path, overwrite=args.overwrite)
        print(f"  {csv_path.name} -> {h5_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
