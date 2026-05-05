from __future__ import annotations

import argparse

import pandas as pd
import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
from scripts.config import RESULTS_DIR
from scripts.features.angle_features import summarize_angle_features_from_ids
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

    angle_saline = summarize_angle_features_from_ids(
        saline_ids,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
    )

    angle_ghrelin = summarize_angle_features_from_ids(
        ghrelin_ids,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
    )

    summary_df = pd.DataFrame(
        {
            "group": (
                ["Saline"] * len(angle_saline)
                + ["Ghrelin"] * len(angle_ghrelin)
            ),
            "head_body_misalignment": angle_saline + angle_ghrelin,
        }
    )

    csv_path = RESULTS_DIR / f"{args.task.lower()}_angle_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    ax = barplot_mean_se(
        angle_saline,
        angle_ghrelin,
        labels=["Saline", "Ghrelin"],
        ylabel=f"{args.how.capitalize()} head-body misalignment (rad)",
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