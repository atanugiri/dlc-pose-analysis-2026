from __future__ import annotations

import os
import re
from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd

# Set paper-specific env before importing project modules that load config.
os.environ["ENV_FILE"] = ".env.paper2026"

from scripts.config import RESULTS_DIR
from scripts.db.get_filtered_ids import get_filtered_ids
from scripts.features.angle_features import head_body_misalignment_metrics_from_ids
from scripts.features.motion_features import summarize_speed_from_ids
from scripts.features.trajectory_curvature import summarize_curvature_from_ids
from scripts.plots.group_comparison_plot import plot_group_comparison


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "group"


def _fetch_task_groups(tasks: list[str]) -> dict[str, tuple[list[int], list[int]]]:
    groups: dict[str, tuple[list[int], list[int]]] = {}
    for task in tasks:
        saline_ids = get_filtered_ids({"task": task, "treatment": "Y"})
        ghrelin_ids = get_filtered_ids({"task": task, "treatment": "P"})
        groups[task] = (saline_ids, ghrelin_ids)
    return groups


def _run_feature_analysis(
    *,
    feature_name: str,
    tasks: list[str],
    groups: dict[str, tuple[list[int], list[int]]],
    compute_values: Callable[[str, list[int]], list[float]],
    value_column: str,
    ylabel: str,
    per_task_title: Callable[[str], str],
    suffix_builder: Callable[[str], str],
    combined_suffix: str,
    plot_type: str = "box",
    test: str = "welch",
) -> tuple[list[str], list[str]]:
    analysis_dir = RESULTS_DIR / f"{feature_name}_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    excel_paths: list[str] = []
    fig_paths: list[str] = []
    combined_rows: list[dict[str, object]] = []
    combined_saline: list[float] = []
    combined_ghrelin: list[float] = []

    for task in tasks:
        saline_ids, ghrelin_ids = groups[task]
        saline_values = compute_values(task, saline_ids)
        ghrelin_values = compute_values(task, ghrelin_ids)
        combined_saline.extend(saline_values)
        combined_ghrelin.extend(ghrelin_values)

        summary_rows = [
            {"id": record_id, "group": "Saline", value_column: value}
            for record_id, value in zip(saline_ids, saline_values)
        ]
        summary_rows.extend(
            {"id": record_id, "group": "Ghrelin", value_column: value}
            for record_id, value in zip(ghrelin_ids, ghrelin_values)
        )
        combined_rows.extend(summary_rows)

        summary_df = pd.DataFrame(summary_rows)
        task_slug = _slugify(task)
        stem = f"{task_slug}_{suffix_builder(task)}"
        excel_path = analysis_dir / f"{stem}_summary.xlsx"
        summary_df.to_excel(excel_path, index=False)

        ax = plot_group_comparison(
            saline_values,
            ghrelin_values,
            labels=["Saline", "Ghrelin"],
            ylabel=ylabel,
            test=test,
            plot_type=plot_type,
        )
        ax.set_title(per_task_title(task))
        plt.tight_layout()
        fig_path = analysis_dir / f"{stem}_{plot_type}plot.pdf"
        plt.savefig(fig_path, dpi=300)
        plt.close()

        excel_paths.append(str(excel_path))
        fig_paths.append(str(fig_path))

    combined_df = pd.DataFrame(combined_rows)
    combined_name = "_".join(_slugify(task) for task in tasks)
    combined_stem = f"{combined_name}_{combined_suffix}"
    combined_excel_path = analysis_dir / f"{combined_stem}_summary.xlsx"
    combined_df.to_excel(combined_excel_path, index=False)

    ax = plot_group_comparison(
        combined_saline,
        combined_ghrelin,
        labels=["Saline", "Ghrelin"],
        ylabel=ylabel,
        test=test,
        plot_type=plot_type,
    )
    ax.set_title(f"{combined_name}: {feature_name}")
    plt.tight_layout()
    combined_fig_path = analysis_dir / f"{combined_stem}_{plot_type}plot.pdf"
    plt.savefig(combined_fig_path, dpi=300)
    plt.close()

    excel_paths.append(str(combined_excel_path))
    fig_paths.append(str(combined_fig_path))

    return excel_paths, fig_paths


def main() -> None:
    print("Fetching IDs for paper2026 tasks...")
    tasks = ["ToyRAT", "ToyStick"]
    groups = _fetch_task_groups(tasks)

    # Task-specific individual override for the ToyRAT recordings.
    speed_individual_by_task = {"ToyRAT": "m1", "ToyStick": None}
    curvature_individual_by_task = {"ToyRAT": "m1", "ToyStick": None}
    angle_individual_by_task = {"ToyRAT": "m1", "ToyStick": None}

    print("Running speed analysis...")
    speed_excels, speed_figs = _run_feature_analysis(
        feature_name="speed",
        tasks=tasks,
        groups=groups,
        compute_values=lambda task, ids: summarize_speed_from_ids(
            ids,
            bodypart="Head",
            how="mean",
            individual=speed_individual_by_task[task],
            smoothing_window=None,
            likelihood_threshold=None,
            normalization=True,
        ),
        value_column="speed",
        ylabel="Mean speed",
        per_task_title=lambda task: f"{_slugify(task)}: Head speed",
        suffix_builder=lambda _task: "head_sw_None_lt_None_speed",
        combined_suffix="head_sw_None_lt_None_speed",
        plot_type="box",
    )

    print("Running curvature analysis...")
    curvature_excels, curvature_figs = _run_feature_analysis(
        feature_name="curvature",
        tasks=tasks,
        groups=groups,
        compute_values=lambda task, ids: summarize_curvature_from_ids(
            ids,
            bodypart="Midback",
            how="mean",
            individual=curvature_individual_by_task[task],
            smoothing_window=5,
            speed_thresh=0.01,
            likelihood_threshold=0.5,
            normalization=False,
        ),
        value_column="curvature",
        ylabel="Mean curvature",
        per_task_title=lambda task: f"{_slugify(task)}: Midback curvature",
        suffix_builder=lambda _task: "mean_midback_sw_5_lt_0.5_st_0.01_curvature",
        combined_suffix="mean_midback_sw_5_lt_0.5_st_0.01_curvature",
        plot_type="box",
    )

    print("Running angle analysis...")
    angle_excels, angle_figs = _run_feature_analysis(
        feature_name="angle",
        tasks=tasks,
        groups=groups,
        compute_values=lambda task, ids: [
            row["median"]
            for row in head_body_misalignment_metrics_from_ids(
                ids,
                likelihood_threshold=0.8,
                individual=angle_individual_by_task[task],
            )
        ],
        value_column="angle",
        ylabel="Head-body misalignment median (rad)",
        per_task_title=lambda task: f"{_slugify(task)}: head-body misalignment median",
        suffix_builder=lambda _task: "lt_0.8_angle",
        combined_suffix="lt_0.8_angle",
        plot_type="box",
    )

    print("Saved speed Excel files:")
    for path in speed_excels:
        print(path)
    print("Saved speed figure files:")
    for path in speed_figs:
        print(path)

    print("Saved curvature Excel files:")
    for path in curvature_excels:
        print(path)
    print("Saved curvature figure files:")
    for path in curvature_figs:
        print(path)

    print("Saved angle Excel files:")
    for path in angle_excels:
        print(path)
    print("Saved angle figure files:")
    for path in angle_figs:
        print(path)


if __name__ == "__main__":
    main()