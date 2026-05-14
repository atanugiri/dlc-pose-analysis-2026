from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
from scripts.config import RESULTS_DIR
from scripts.features.motion_features import summarize_speed_from_ids
from scripts.plots.feature_barplot import barplot_mean_se


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run speed analysis for ChickenBroth saline vs ghrelin."
    )

    parser.add_argument("--task", default="ChickenBroth")
    parser.add_argument("--bodypart", default="Head")
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
        help="Optional smoothing window size for trajectory smoothing.",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=None,
        help="Likelihood threshold for filtering low-confidence poses.",
    )
    parser.add_argument(
        "--normalization",
        type=bool,
        default=True,
        help="Whether to normalize coordinates to unit square using maze corners.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    speed_analysis_dir = RESULTS_DIR / "speed_analysis"
    speed_analysis_dir.mkdir(exist_ok=True)

    saline_ids = db_utils.get_treatment_ids(args.task, 'Y')
    ghrelin_ids = db_utils.get_treatment_ids(args.task, 'P')

    print(f"{args.task}-Saline IDs: {len(saline_ids)}")
    print(f"{args.task}-Ghrelin IDs: {len(ghrelin_ids)}")

    speed_saline = summarize_speed_from_ids(
        saline_ids,
        bodypart=args.bodypart,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
        normalization=args.normalization,
    )

    speed_ghrelin = summarize_speed_from_ids(
        ghrelin_ids,
        bodypart=args.bodypart,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
        normalization=args.normalization,
    )

    summary_df = pd.DataFrame(
        {
            "group": (
                ["Saline"] * len(speed_saline)
                + ["Ghrelin"] * len(speed_ghrelin)
            ),
            "speed": speed_saline + speed_ghrelin,
        }
    )

    csv_path = speed_analysis_dir / f"{args.task.lower()}_{args.bodypart.lower()}_sw_{args.smoothing_window}_lt_{args.likelihood_threshold}_speed_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    ax = barplot_mean_se(
        speed_saline,
        speed_ghrelin,
        labels=["Saline", "Ghrelin"],
        ylabel=f"{args.how.capitalize()} speed",
    )

    ax.set_title(f"{args.task}: {args.bodypart} speed")
    plt.tight_layout()

    fig_path = speed_analysis_dir / f"{args.task.lower()}_{args.bodypart.lower()}_sw_{args.smoothing_window}_lt_{args.likelihood_threshold}_speed_barplot.png"
    plt.savefig(fig_path, dpi=300)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()