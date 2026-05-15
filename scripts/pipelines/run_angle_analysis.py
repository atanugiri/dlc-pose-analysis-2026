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
        description="Run angle analysis for specified task(s), saline vs ghrelin."
    )

    parser.add_argument("--task", nargs='+', default=["ChickenBroth"], help="Task name(s) to analyze (e.g., --task ToyRAT ToyStick)")
    parser.add_argument(
        "--individual",
        default=None,
        help="Optional individual name for multi-animal files (e.g. 'm1').",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=None,
        help="Likelihood threshold for filtering low-confidence poses.",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    angle_analysis_dir = RESULTS_DIR / "angle_analysis"
    angle_analysis_dir.mkdir(exist_ok=True)

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

    angle_saline = head_body_misalignment_p95_from_ids(
        saline_ids,
        likelihood_threshold=args.likelihood_threshold,
        individual=args.individual,
    )

    angle_ghrelin = head_body_misalignment_p95_from_ids(
        ghrelin_ids,
        likelihood_threshold=args.likelihood_threshold,
        individual=args.individual,
    )

    summary_df = pd.DataFrame(
        {
            "group": (
                ["Saline"] * len(angle_saline)
                + ["Ghrelin"] * len(angle_ghrelin)
            ),
            "head_body_misalignment_p95": angle_saline + angle_ghrelin,
        }
    )

    csv_path = angle_analysis_dir / f"{task_name.lower()}_lt_{args.likelihood_threshold}_angle_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    ax = barplot_mean_se(
        angle_saline,
        angle_ghrelin,
        labels=["Saline", "Ghrelin"],
        ylabel="Head-body misalignment p95 (rad)",
    )

    ax.set_title(f"{task_name}: head-body misalignment")
    plt.tight_layout()

    fig_path = angle_analysis_dir / f"{task_name.lower()}_lt_{args.likelihood_threshold}_angle_barplot.png"
    plt.savefig(fig_path, dpi=300)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()