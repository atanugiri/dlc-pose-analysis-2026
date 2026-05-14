from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
from scripts.config import RESULTS_DIR
from scripts.features.trajectory_curvature import summarize_curvature_from_ids
from scripts.plots.feature_barplot import barplot_mean_se


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run curvature analysis for ChickenBroth saline vs ghrelin."
    )

    parser.add_argument("--task", default="ChickenBroth")
    parser.add_argument("--bodypart", default="Midback")
    parser.add_argument(
        "--individual",
        default=None,
        help="Optional individual name for multi-animal files (e.g. 'm1').",
    )
    parser.add_argument("--how", default="mean", choices=["mean", "median", "max", "std"])
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Smoothing window size for trajectory smoothing.",
    )
    parser.add_argument(
        "--speed-thresh",
        type=float,
        default=0.01,
        help="Speed threshold for filtering low-speed frames.",
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

    curvature_saline = summarize_curvature_from_ids(
        saline_ids,
        bodypart=args.bodypart,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        speed_thresh=args.speed_thresh,
        likelihood_threshold=args.likelihood_threshold,
    )

    curvature_ghrelin = summarize_curvature_from_ids(
        ghrelin_ids,
        bodypart=args.bodypart,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        speed_thresh=args.speed_thresh,
        likelihood_threshold=args.likelihood_threshold,
    )

    summary_df = pd.DataFrame(
        {
            "group": (
                ["Saline"] * len(curvature_saline)
                + ["Ghrelin"] * len(curvature_ghrelin)
            ),
            "curvature": curvature_saline + curvature_ghrelin,
        }
    )

    csv_path = RESULTS_DIR / f"{args.task.lower()}_curvature_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    ax = barplot_mean_se(
        curvature_saline,
        curvature_ghrelin,
        labels=["Saline", "Ghrelin"],
        ylabel=f"{args.how.capitalize()} curvature",
    )

    ax.set_title(f"{args.task}: {args.bodypart} curvature")
    plt.tight_layout()

    fig_path = RESULTS_DIR / f"{args.task.lower()}_curvature_barplot.png"
    plt.savefig(fig_path, dpi=300)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()