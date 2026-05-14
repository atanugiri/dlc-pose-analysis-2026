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
        description="Run curvature analysis for specified task(s), saline vs ghrelin."
    )

    parser.add_argument("--task", nargs='+', default=["ChickenBroth"], help="Task name(s) to analyze (e.g., --task ToyRAT ToyStick)")
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
        default=None,
        help="Likelihood threshold for filtering low-confidence poses.",
    )
    parser.add_argument(
        "--normalization",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Whether to normalize coordinates (true/false).",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    curvature_analysis_dir = RESULTS_DIR / "curvature_analysis"
    curvature_analysis_dir.mkdir(exist_ok=True)

    # Combine IDs from all specified tasks
    saline_ids = []
    ghrelin_ids = []
    for task in args.task:
        saline_ids.extend(db_utils.get_treatment_ids(task, 'Y'))
        ghrelin_ids.extend(db_utils.get_treatment_ids(task, 'P'))

    task_name = "_".join(args.task)
    print(f"Tasks: {task_name}")
    print(f"Saline IDs: {len(saline_ids)}")
    print(f"Ghrelin IDs: {len(ghrelin_ids)}")

    curvature_saline = summarize_curvature_from_ids(
        saline_ids,
        bodypart=args.bodypart,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        speed_thresh=args.speed_thresh,
        likelihood_threshold=args.likelihood_threshold,
        normalization=args.normalization,
    )

    curvature_ghrelin = summarize_curvature_from_ids(
        ghrelin_ids,
        bodypart=args.bodypart,
        how=args.how,
        individual=args.individual,
        smoothing_window=args.smoothing_window,
        speed_thresh=args.speed_thresh,
        likelihood_threshold=args.likelihood_threshold,
        normalization=args.normalization,
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

    csv_path = curvature_analysis_dir / f"{task_name.lower()}_{args.how}_{args.bodypart.lower()}_sw_{args.smoothing_window}_lt_{args.likelihood_threshold}_st_{args.speed_thresh}_curvature_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    ax = barplot_mean_se(
        curvature_saline,
        curvature_ghrelin,
        labels=["Saline", "Ghrelin"],
        ylabel=f"{args.how.capitalize()} curvature",
    )

    ax.set_title(f"{task_name}: {args.bodypart} curvature")
    plt.tight_layout()

    fig_path = curvature_analysis_dir / f"{task_name.lower()}_{args.how}_{args.bodypart.lower()}_sw_{args.smoothing_window}_lt_{args.likelihood_threshold}_st_{args.speed_thresh}_curvature_barplot.png"
    plt.savefig(fig_path, dpi=300)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()