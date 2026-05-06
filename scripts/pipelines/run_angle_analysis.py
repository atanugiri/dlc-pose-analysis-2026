from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
from scripts.config import RESULTS_DIR
from scripts.features.angle_features import batch_angle_features
from scripts.plots.feature_barplot import barplot_mean_se


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run angle analysis for ChickenBroth saline vs ghrelin."
    )

    parser.add_argument("--task", default="ChickenBroth")
    parser.add_argument(
        "--individual",
        default=None,
        help="Optional individual name for multi-animal files (e.g. 'm1').",
    )
    parser.add_argument("--how", default="mean", choices=["mean", "median", "max", "std"])
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        help="Optional smoothing window size for head-body misalignment.",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=0.5,
        help="Likelihood threshold for filtering low-confidence poses.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    query_saline = """
        SELECT id
        FROM public.experimental_metadata
        WHERE task = %s
          AND treatment = 'Y'
        ORDER BY id;
    """

    query_ghrelin = """
        SELECT id
        FROM public.experimental_metadata
        WHERE task = %s
          AND treatment = 'P'
        ORDER BY id;
    """

    saline_ids = db_utils.fetch_ids_with_params(query_saline, (args.task,))
    ghrelin_ids = db_utils.fetch_ids_with_params(query_ghrelin, (args.task,))

    print(f"{args.task}-Saline IDs: {len(saline_ids)}")
    print(f"{args.task}-Ghrelin IDs: {len(ghrelin_ids)}")

    # Compute full summary statistics for each group
    angle_saline = batch_angle_features(
        None,  # dlc_table (unused, for API compatibility)
        saline_ids,
        likelihood_threshold=args.likelihood_threshold,
        smooth_window=args.smoothing_window,
    )

    angle_ghrelin = batch_angle_features(
        None,  # dlc_table (unused, for API compatibility)
        ghrelin_ids,
        likelihood_threshold=args.likelihood_threshold,
        smooth_window=args.smoothing_window,
    )

    # Combine results and add group label
    angle_saline['group'] = 'Saline'
    angle_ghrelin['group'] = 'Ghrelin'
    summary_df = pd.concat([angle_saline, angle_ghrelin], ignore_index=True)

    # Save summary to CSV
    csv_path = RESULTS_DIR / f"{args.task.lower()}_angle_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved {csv_path}")

    # Plot head-body misalignment by group
    saline_vals = angle_saline['head_body_misalignment_mean'].dropna()
    ghrelin_vals = angle_ghrelin['head_body_misalignment_mean'].dropna()

    ax = barplot_mean_se(
        saline_vals,
        ghrelin_vals,
        labels=["Saline", "Ghrelin"],
        ylabel="Mean head-body misalignment (rad)",
    )

    ax.set_title(f"{args.task}: head-body misalignment")
    plt.tight_layout()

    fig_path = RESULTS_DIR / f"{args.task.lower()}_angle_barplot.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()