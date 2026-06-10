from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.config import RESULTS_DIR
from scripts.plots.group_comparison_plot import plot_group_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple analysis Excel files and generate combined plot."
    )
    parser.add_argument(
        "excel_files",
        nargs="+",
        type=Path,
        help="Excel files to combine (e.g., toyrat_head_*.xlsx toystick_head_*.xlsx)",
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
        help="Feature column name (default: 'speed', options: 'speed', 'curvature', 'angle')",
    )
    parser.add_argument(
        "--test",
        choices=["welch", "mann_whitney"],
        default="welch",
        help="Statistical test to use (welch=two-tailed t-test, mann_whitney=non-parametric).",
    )
    parser.add_argument(
        "--plot-type",
        choices=["bar", "box"],
        default="bar",
        help="Plot style to generate.",
    )

    args = parser.parse_args()

    for f in args.excel_files:
        if not f.exists():
            raise FileNotFoundError(f"Excel file not found: {f}")

    dfs = [pd.read_excel(f) for f in args.excel_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Combined {len(args.excel_files)} files:")
    for f in args.excel_files:
        print(f"  - {f.name}")
    print(f"Total rows: {len(combined_df)}")

    feature = args.feature
    if feature not in combined_df.columns:
        raise ValueError(
            f"Feature '{feature}' not found in CSV. "
            f"Available columns: {combined_df.columns.tolist()}"
        )

    saline_values = combined_df[combined_df["group"] == "Saline"][feature].tolist()
    ghrelin_values = combined_df[combined_df["group"] == "Ghrelin"][feature].tolist()

    print(f"Saline samples: {len(saline_values)}")
    print(f"Ghrelin samples: {len(ghrelin_values)}")

    ax = plot_group_comparison(
        saline_values,
        ghrelin_values,
        labels=["Saline", "Ghrelin"],
        ylabel=feature,
        show_points=True,
        test=args.test,
        plot_type=args.plot_type,
    )

    ax.set_title(f"Combined tasks: {feature}")
    plt.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    feature_dir_map = {
        "speed": "speed_analysis",
        "curvature": "curvature_analysis",
        "angle": "angle_analysis",
    }
    analysis_subdir = feature_dir_map.get(feature, "analysis")
    analysis_dir = RESULTS_DIR / analysis_subdir
    analysis_dir.mkdir(exist_ok=True)

    excel_path = analysis_dir / f"{args.output_name}_{feature}_summary.xlsx"
    combined_df.to_excel(excel_path, index=False)

    fig_path = analysis_dir / f"{args.output_name}_{feature}_{args.plot_type}plot.pdf"
    plt.savefig(fig_path, dpi=300)

    print(f"Saved combined Excel: {excel_path}")
    print(f"Saved combined plot: {fig_path}")


if __name__ == "__main__":
    main()