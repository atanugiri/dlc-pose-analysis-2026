from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
from scripts.config import RESULTS_DIR
from scripts.features.angle_features import head_body_misalignment_p95_from_ids
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

    saline_ids = db_utils.get_treatment_ids(args.task, 'Y')
    ghrelin_ids = db_utils.get_treatment_ids(args.task, 'P')

    print(f"{args.task}-Saline IDs: {len(saline_ids)}")
    print(f"{args.task}-Ghrelin IDs: {len(ghrelin_ids)}")

    # Compute head-body misalignment p95 for each group
    saline_p95 = head_body_misalignment_p95_from_ids(
        saline_ids,
        likelihood_threshold=args.likelihood_threshold,
        individual=args.individual,
    )

    ghrelin_p95 = head_body_misalignment_p95_from_ids(
        ghrelin_ids,
        likelihood_threshold=args.likelihood_threshold,
        individual=args.individual,
    )

    angle_saline = pd.DataFrame({"trial_id": saline_ids, "head_body_misalignment_p95": saline_p95})
    angle_ghrelin = pd.DataFrame({"trial_id": ghrelin_ids, "head_body_misalignment_p95": ghrelin_p95})

    # Combine results and add group label
    angle_saline["group"] = "Saline"
    angle_ghrelin["group"] = "Ghrelin"
    summary_df = pd.concat([angle_saline, angle_ghrelin], ignore_index=True)

    # Save summary to CSV
    csv_path = RESULTS_DIR / f"{args.task.lower()}_angle_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"[✓] Saved {csv_path}")

    # Plot head-body misalignment p95 by group
    saline_vals = angle_saline['head_body_misalignment_p95'].dropna()
    ghrelin_vals = angle_ghrelin['head_body_misalignment_p95'].dropna()

    ax = barplot_mean_se(
        saline_vals,
        ghrelin_vals,
        labels=["Saline", "Ghrelin"],
        ylabel="Head-body misalignment p95 (rad)",
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