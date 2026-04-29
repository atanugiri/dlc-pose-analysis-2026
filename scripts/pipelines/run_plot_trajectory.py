from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

import scripts.db.db_utils as db_utils
from scripts.config import RESULTS_DIR
from scripts.plots.plot_trajectory import plot_trajectory_from_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trajectories for a task (Saline vs Ghrelin).")

    parser.add_argument("--task", default="ChickenBroth")
    parser.add_argument("--bodypart", default="Midback")
    parser.add_argument(
        "--individual",
        default=None,
        help="Optional individual name for multi-animal files (e.g. 'm1').",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=0.2,
        help="Line width for plotted trajectories.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Alpha transparency for plotted trajectories.",
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

    # Saline
    if saline_ids:
        plot_trajectory_from_ids(
            saline_ids,
            bodypart=args.bodypart,
            individual=args.individual,
            linewidth=args.linewidth,
            alpha=args.alpha,
        )
        fig_path = RESULTS_DIR / f"{args.task.lower()}_saline_trajectories.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved figure: {fig_path}")

    # Ghrelin
    if ghrelin_ids:
        plot_trajectory_from_ids(
            ghrelin_ids,
            bodypart=args.bodypart,
            individual=args.individual,
            linewidth=args.linewidth,
            alpha=args.alpha,
        )
        fig_path = RESULTS_DIR / f"{args.task.lower()}_ghrelin_trajectories.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
