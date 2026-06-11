from __future__ import annotations

import os
import re
import subprocess
import sys

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


def _save_summary_and_plot(
    *,
    analysis_dir,
    stem: str,
    summary_df: pd.DataFrame,
    saline_values: list[float],
    ghrelin_values: list[float],
    ylabel: str,
    title: str,
    plot_type: str = "box",
    test: str = "welch",
) -> tuple[str, str]:
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
    ax.set_title(title)
    plt.tight_layout()
    fig_path = analysis_dir / f"{stem}_{plot_type}plot.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return str(excel_path), str(fig_path)


def _combine_task_excels(
    *,
    excel_paths: list[str],
    feature: str,
    output_name: str = "toyrat_toystick",
    plot_type: str = "box",
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.utils.combine_task_analysis",
            *excel_paths,
            "--output-name",
            output_name,
            "--feature",
            feature,
            "--plot-type",
            plot_type,
        ],
        check=True,
    )


def _run_speed_analysis(
    tasks: list[str],
    groups: dict[str, tuple[list[int], list[int]]],
) -> tuple[list[str], list[str]]:
    analysis_dir = RESULTS_DIR / "speed_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    individual_by_task = {"ToyRAT": "m1", "ToyStick": None}

    excel_paths: list[str] = []
    fig_paths: list[str] = []

    for task in tasks:
        saline_ids, ghrelin_ids = groups[task]
        saline_values = summarize_speed_from_ids(
            saline_ids,
            bodypart="Head",
            how="mean",
            individual=individual_by_task[task],
            smoothing_window=None,
            likelihood_threshold=None,
            normalization=True,
        )
        ghrelin_values = summarize_speed_from_ids(
            ghrelin_ids,
            bodypart="Head",
            how="mean",
            individual=individual_by_task[task],
            smoothing_window=None,
            likelihood_threshold=None,
            normalization=True,
        )

        summary_rows = [
            {"id": record_id, "group": "Saline", "speed": value}
            for record_id, value in zip(saline_ids, saline_values)
        ]
        summary_rows.extend(
            {"id": record_id, "group": "Ghrelin", "speed": value}
            for record_id, value in zip(ghrelin_ids, ghrelin_values)
        )

        task_slug = _slugify(task)
        stem = f"{task_slug}_head_sw_None_lt_None_speed"
        excel_path, fig_path = _save_summary_and_plot(
            analysis_dir=analysis_dir,
            stem=stem,
            summary_df=pd.DataFrame(summary_rows),
            saline_values=saline_values,
            ghrelin_values=ghrelin_values,
            ylabel="Mean speed",
            title=f"{task_slug}: Head speed",
            plot_type="box",
        )
        excel_paths.append(excel_path)
        fig_paths.append(fig_path)

    return excel_paths, fig_paths


def _run_curvature_analysis(
    tasks: list[str],
    groups: dict[str, tuple[list[int], list[int]]],
) -> tuple[list[str], list[str]]:
    analysis_dir = RESULTS_DIR / "curvature_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    individual_by_task = {"ToyRAT": "m1", "ToyStick": None}

    excel_paths: list[str] = []
    fig_paths: list[str] = []

    for task in tasks:
        saline_ids, ghrelin_ids = groups[task]
        saline_values = summarize_curvature_from_ids(
            saline_ids,
            bodypart="Midback",
            how="mean",
            individual=individual_by_task[task],
            smoothing_window=5,
            speed_thresh=0.01,
            likelihood_threshold=0.5,
            normalization=False,
        )
        ghrelin_values = summarize_curvature_from_ids(
            ghrelin_ids,
            bodypart="Midback",
            how="mean",
            individual=individual_by_task[task],
            smoothing_window=5,
            speed_thresh=0.01,
            likelihood_threshold=0.5,
            normalization=False,
        )

        summary_rows = [
            {"id": record_id, "group": "Saline", "curvature": value}
            for record_id, value in zip(saline_ids, saline_values)
        ]
        summary_rows.extend(
            {"id": record_id, "group": "Ghrelin", "curvature": value}
            for record_id, value in zip(ghrelin_ids, ghrelin_values)
        )

        task_slug = _slugify(task)
        stem = f"{task_slug}_mean_midback_sw_5_lt_0.5_st_0.01_curvature"
        excel_path, fig_path = _save_summary_and_plot(
            analysis_dir=analysis_dir,
            stem=stem,
            summary_df=pd.DataFrame(summary_rows),
            saline_values=saline_values,
            ghrelin_values=ghrelin_values,
            ylabel="Mean curvature",
            title=f"{task_slug}: Midback curvature",
            plot_type="box",
        )
        excel_paths.append(excel_path)
        fig_paths.append(fig_path)

    return excel_paths, fig_paths


def _run_angle_analysis(
    tasks: list[str],
    groups: dict[str, tuple[list[int], list[int]]],
) -> tuple[list[str], list[str]]:
    analysis_dir = RESULTS_DIR / "angle_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    individual_by_task = {"ToyRAT": "m1", "ToyStick": None}

    excel_paths: list[str] = []
    fig_paths: list[str] = []

    for task in tasks:
        saline_ids, ghrelin_ids = groups[task]
        saline_values = [
            row["median"]
            for row in head_body_misalignment_metrics_from_ids(
                saline_ids,
                likelihood_threshold=0.8,
                individual=individual_by_task[task],
            )
        ]
        ghrelin_values = [
            row["median"]
            for row in head_body_misalignment_metrics_from_ids(
                ghrelin_ids,
                likelihood_threshold=0.8,
                individual=individual_by_task[task],
            )
        ]

        summary_rows = [
            {"id": record_id, "group": "Saline", "angle": value}
            for record_id, value in zip(saline_ids, saline_values)
        ]
        summary_rows.extend(
            {"id": record_id, "group": "Ghrelin", "angle": value}
            for record_id, value in zip(ghrelin_ids, ghrelin_values)
        )

        task_slug = _slugify(task)
        stem = f"{task_slug}_lt_0.8_angle"
        excel_path, fig_path = _save_summary_and_plot(
            analysis_dir=analysis_dir,
            stem=stem,
            summary_df=pd.DataFrame(summary_rows),
            saline_values=saline_values,
            ghrelin_values=ghrelin_values,
            ylabel="Head-body misalignment median (rad)",
            title=f"{task_slug}: head-body misalignment median",
            plot_type="box",
        )
        excel_paths.append(excel_path)
        fig_paths.append(fig_path)

    return excel_paths, fig_paths


def main() -> None:
    print("Fetching IDs for paper2026 tasks...")
    tasks = ["ToyRAT", "ToyStick"]
    groups = _fetch_task_groups(tasks)

    print("Running speed analysis...")
    speed_excels, speed_figs = _run_speed_analysis(tasks, groups)
    print("Combining speed task summaries...")
    _combine_task_excels(excel_paths=speed_excels, feature="speed", plot_type="box")

    print("Running curvature analysis...")
    curvature_excels, curvature_figs = _run_curvature_analysis(tasks, groups)
    print("Combining curvature task summaries...")
    _combine_task_excels(excel_paths=curvature_excels, feature="curvature", plot_type="box")

    print("Running angle analysis...")
    angle_excels, angle_figs = _run_angle_analysis(tasks, groups)
    print("Combining angle task summaries...")
    _combine_task_excels(excel_paths=angle_excels, feature="angle", plot_type="box")


if __name__ == "__main__":
    main()