from __future__ import annotations

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt

from scripts.config import RESULTS_DIR
from scripts.features.motion_features import summarize_speed_from_ids
from scripts.plots.group_comparison_plot import plot_group_comparison


def _parse_id_list(value: str) -> list[int]:
    """Parse a comma-separated list of integer IDs."""
    try:
        ids = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ID list: {value!r}") from exc

    if not ids:
        raise argparse.ArgumentTypeError("Each --id-list must contain at least one integer ID.")

    return ids


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "group"


def run_speed_analysis_groups(
    *,
    id_lists: list[list[int]],
    labels: list[str],
    analysis_name: str | None = None,
    bodypart: str = "Head",
    individual: str | None = None,
    how: str = "mean",
    smoothing_window: int | None = None,
    likelihood_threshold: float | None = None,
    normalization: bool = True,
    test: str = "welch",
    plot_type: str = "bar",
) -> tuple[pd.DataFrame, str, str]:
    """Run speed analysis from explicit groups and return outputs.

    Returns:
        tuple(summary_df, excel_path, fig_path)
    """
    if len(id_lists) != len(labels):
        raise ValueError("Number of id_lists and labels must match.")
    if len(id_lists) < 2:
        raise ValueError("Provide at least two groups for comparison.")

    RESULTS_DIR.mkdir(exist_ok=True)
    speed_analysis_dir = RESULTS_DIR / "speed_analysis"
    speed_analysis_dir.mkdir(exist_ok=True)

    out_name = (
        _slugify(analysis_name)
        if analysis_name
        else "_".join(_slugify(label) for label in labels)
    )

    all_speeds: list[list[float]] = []
    summary_rows: list[dict[str, object]] = []
    for label, record_ids in zip(labels, id_lists):
        speeds = summarize_speed_from_ids(
            record_ids,
            bodypart=bodypart,
            how=how,
            individual=individual,
            smoothing_window=smoothing_window,
            likelihood_threshold=likelihood_threshold,
            normalization=normalization,
        )
        all_speeds.append(speeds)
        summary_rows.extend(
            {
                "id": record_id,
                "group": label,
                "speed": speed,
            }
            for record_id, speed in zip(record_ids, speeds)
        )

    summary_df = pd.DataFrame(summary_rows)

    excel_path = speed_analysis_dir / f"{out_name}_{bodypart.lower()}_sw_{smoothing_window}_lt_{likelihood_threshold}_speed_summary.xlsx"
    summary_df.to_excel(excel_path, index=False)

    ax = plot_group_comparison(
        *all_speeds,
        labels=labels,
        ylabel=f"{how.capitalize()} speed",
        test=test,
        plot_type=plot_type,
    )

    ax.set_title(f"{out_name}: {bodypart} speed")
    plt.tight_layout()

    fig_path = speed_analysis_dir / f"{out_name}_{bodypart.lower()}_sw_{smoothing_window}_lt_{likelihood_threshold}_speed_{plot_type}plot.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return summary_df, str(excel_path), str(fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run speed analysis from explicit ID groups and labels."
    )

    parser.add_argument(
        "--analysis-name",
        default=None,
        help="Optional output name prefix. Defaults to slugified labels.",
    )
    parser.add_argument(
        "--id-list",
        action="append",
        type=_parse_id_list,
        required=True,
        help="Comma-separated IDs for one group (repeat for multiple groups).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Label for each --id-list group (repeat; order must match --id-list).",
    )
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
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Whether to normalize coordinates (true/false).",
    )
    parser.add_argument(
        "--test",
        choices=['welch', 'mann_whitney'],
        default='welch',
        help="Statistical test to use (welch=two-tailed t-test, mann_whitney=non-parametric).",
    )
    parser.add_argument(
        "--plot-type",
        choices=["bar", "box"],
        default="bar",
        help="Plot style to generate.",
    )

    args = parser.parse_args()

    if args.label is None:
        parser.error("Provide --label for each --id-list.")
    if len(args.id_list) != len(args.label):
        parser.error("Number of --id-list and --label arguments must match.")

    _, excel_path, fig_path = run_speed_analysis_groups(
        id_lists=args.id_list,
        labels=args.label,
        analysis_name=args.analysis_name,
        bodypart=args.bodypart,
        individual=args.individual,
        how=args.how,
        smoothing_window=args.smoothing_window,
        likelihood_threshold=args.likelihood_threshold,
        normalization=args.normalization,
        test=args.test,
        plot_type=args.plot_type,
    )

    print(f"Saved Excel: {excel_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()