from __future__ import annotations

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt

from scripts.config import RESULTS_DIR
from scripts.features.angle_features import head_body_misalignment_metrics_from_ids
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


def run_angle_analysis_groups(
    *,
    id_lists: list[list[int]],
    labels: list[str],
    analysis_name: str | None = None,
    individual: str | None = None,
    likelihood_threshold: float | None = None,
    metric: str = "p95",
    test: str = "welch",
    plot_type: str = "bar",
) -> tuple[pd.DataFrame, str, str]:
    """Run angle analysis from explicit groups and return outputs.

    Returns:
        tuple(summary_df, excel_path, fig_path)
    """
    if len(id_lists) != len(labels):
        raise ValueError("Number of id_lists and labels must match.")
    if len(id_lists) < 2:
        raise ValueError("Provide at least two groups for comparison.")

    RESULTS_DIR.mkdir(exist_ok=True)
    angle_analysis_dir = RESULTS_DIR / "angle_analysis"
    angle_analysis_dir.mkdir(exist_ok=True)

    out_name = (
        _slugify(analysis_name)
        if analysis_name
        else "_".join(_slugify(label) for label in labels)
    )

    all_angles: list[list[float]] = []
    summary_rows: list[dict[str, object]] = []

    for label, record_ids in zip(labels, id_lists):
        angle_dicts = head_body_misalignment_metrics_from_ids(
            record_ids,
            likelihood_threshold=likelihood_threshold,
            individual=individual,
        )
        angles = [d[metric] for d in angle_dicts]
        all_angles.append(angles)
        summary_rows.extend(
            {
                "id": record_id,
                "group": label,
                "angle": angle,
            }
            for record_id, angle in zip(record_ids, angles)
        )

    summary_df = pd.DataFrame(summary_rows)

    excel_path = angle_analysis_dir / f"{out_name}_lt_{likelihood_threshold}_angle_summary.xlsx"
    summary_df.to_excel(excel_path, index=False)

    ax = plot_group_comparison(
        *all_angles,
        labels=labels,
        ylabel=f"Head-body misalignment {metric} (rad)",
        test=test,
        plot_type=plot_type,
    )

    ax.set_title(f"{out_name}: head-body misalignment {metric}")
    plt.tight_layout()

    fig_path = angle_analysis_dir / f"{out_name}_lt_{likelihood_threshold}_angle_{plot_type}plot.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return summary_df, str(excel_path), str(fig_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run angle analysis from explicit ID groups and labels."
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
    parser.add_argument(
        "--metric",
        choices=['p95', 'mean', 'median', 'max'],
        default='p95',
        help="Which metric to analyze and plot.",
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

    _, excel_path, fig_path = run_angle_analysis_groups(
        id_lists=args.id_list,
        labels=args.label,
        analysis_name=args.analysis_name,
        individual=args.individual,
        likelihood_threshold=args.likelihood_threshold,
        metric=args.metric,
        test=args.test,
        plot_type=args.plot_type,
    )

    print(f"Saved Excel: {excel_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
