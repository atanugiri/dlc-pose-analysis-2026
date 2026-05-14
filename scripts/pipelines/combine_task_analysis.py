from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from scripts.config import RESULTS_DIR
from scripts.plots.feature_barplot import barplot_mean_se


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple analysis CSVs and generate combined plot."
    )
    parser.add_argument(
        "csv_files",
        nargs='+',
        type=Path,
        help="CSV files to combine (e.g., toyrat_head_*.csv toystick_head_*.csv)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="combined",
        help="Name for output files (default: 'combined')",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="speed",
        help="Feature column name (default: 'speed', options: 'speed', 'curvature', 'head_body_misalignment_p95')",
    )

    args = parser.parse_args()

    # Validate files exist
    for f in args.csv_files:
        if not f.exists():
            raise FileNotFoundError(f"CSV file not found: {f}")

    # Read and concatenate all CSVs
    dfs = [pd.read_csv(f) for f in args.csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Combined {len(args.csv_files)} files:")
    for f in args.csv_files:
        print(f"  - {f.name}")
    print(f"Total rows: {len(combined_df)}")

    # Extract saline and ghrelin groups
    feature = args.feature
    if feature not in combined_df.columns:
        raise ValueError(f"Feature '{feature}' not found in CSV. Available columns: {combined_df.columns.tolist()}")
    
    saline_values = combined_df[combined_df["group"] == "Saline"][feature].tolist()
    ghrelin_values = combined_df[combined_df["group"] == "Ghrelin"][feature].tolist()

    print(f"Saline samples: {len(saline_values)}")
    print(f"Ghrelin samples: {len(ghrelin_values)}")

    # Create plot
    ax = barplot_mean_se(
        saline_values,
        ghrelin_values,
        labels=["Saline", "Ghrelin"],
        ylabel=f"Mean ± SE {feature}",
    )

    ax.set_title(f"Combined tasks: {feature}")
    plt.tight_layout()

    # Save combined CSV and plot
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Map feature to analysis directory
    feature_dir_map = {
        "speed": "speed_analysis",
        "curvature": "curvature_analysis",
        "head_body_misalignment_p95": "angle_analysis",
    }
    analysis_subdir = feature_dir_map.get(feature, "analysis")
    analysis_dir = RESULTS_DIR / analysis_subdir
    analysis_dir.mkdir(exist_ok=True)

    csv_path = analysis_dir / f"{args.output_name}_{feature}_summary.csv"
    combined_df.to_csv(csv_path, index=False)

    fig_path = analysis_dir / f"{args.output_name}_{feature}_barplot.png"
    plt.savefig(fig_path, dpi=300)

    print(f"Saved combined CSV: {csv_path}")
    print(f"Saved combined plot: {fig_path}")


if __name__ == "__main__":
    main()
