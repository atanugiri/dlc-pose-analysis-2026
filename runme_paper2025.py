from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

TASKS = ["FoodOnly", "ToyOnly", "LightOnly", "FoodLight", "ToyLight"]


def _force_env_from_file(env_file: str) -> None:
    """Load env file values and force them into process env."""
    env_path = Path(__file__).resolve().parent / env_file
    file_vars = dotenv_values(env_path)
    for key, value in file_vars.items():
        if value is None:
            continue
        os.environ[key] = value


def _fetch_task_groups(
    *,
    dose_mult: int,
    require_modulation_null: bool,
) -> list[tuple[str, list[int], list[int]]]:
    from scripts.db.get_filtered_ids import get_filtered_ids

    grouped: list[tuple[str, list[int], list[int]]] = []
    for task in TASKS:
        base_saline = {"task": task, "treatment": "Y", "dose_mult": dose_mult}
        base_ghrelin = {"task": task, "treatment": "P", "dose_mult": dose_mult}

        if require_modulation_null:
            base_saline["modulation"] = None
            base_ghrelin["modulation"] = None

        saline_ids = get_filtered_ids(base_saline)
        ghrelin_ids = get_filtered_ids(base_ghrelin)
        grouped.append((task, saline_ids, ghrelin_ids))

    return grouped


def _combine_task_excels(*, excel_paths: list[str], output_name: str, feature: str) -> None:
    if len(excel_paths) < 2:
        print(f"Not enough task outputs to combine for {output_name}.")
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.pipelines.combine_task_analysis",
            *excel_paths,
            "--output-name",
            output_name,
            "--feature",
            feature,
            "--plot-type",
            "bar",
        ],
        check=True,
    )


def run_speed_sections() -> None:
    from scripts.pipelines.run_speed_analysis import run_speed_analysis_groups

    sections = [
        # ("10x_ibu", 10, False),
        ("2x_ibu_modulation_null", 2, True),
    ]

    for section_name, dose_mult, require_modulation_null in sections:
        print(f"Running speed analysis: {section_name}")
        grouped = _fetch_task_groups(
            dose_mult=dose_mult,
            require_modulation_null=require_modulation_null,
        )

        excel_paths: list[str] = []
        for task, saline_ids, ghrelin_ids in grouped:
            print(f"{task} | Saline IDs ({section_name}): {len(saline_ids)}")
            print(f"{task} | Ghrelin IDs ({section_name}): {len(ghrelin_ids)}")

            if not saline_ids or not ghrelin_ids:
                print(f"Skipping {task}: one or both groups are empty.")
                continue

            _, excel_path, fig_path = run_speed_analysis_groups(
                id_lists=[saline_ids, ghrelin_ids],
                labels=["Saline", "Ghrelin"],
                analysis_name=f"paper2025_{section_name}_{task}",
                bodypart="Head",
                plot_type="bar",
            )
            excel_paths.append(excel_path)
            print(f"Saved {task} speed Excel: {excel_path}")
            print(f"Saved {task} speed figure: {fig_path}")

        print(f"Combining speed summaries: {section_name}")
        _combine_task_excels(
            excel_paths=excel_paths,
            output_name=f"paper2025_{section_name}_alltasks",
            feature="speed",
        )


# def run_curvature_sections() -> None:
#     from scripts.pipelines.run_curvature_analysis import run_curvature_analysis_groups

#     sections = [
#         ("10x_ibu", 10, False),
#         ("2x_ibu_modulation_null", 2, True),
#     ]

#     for section_name, dose_mult, require_modulation_null in sections:
#         print(f"Running curvature analysis: {section_name}")
#         grouped = _fetch_task_groups(
#             dose_mult=dose_mult,
#             require_modulation_null=require_modulation_null,
#         )

#         excel_paths: list[str] = []
#         for task, saline_ids, ghrelin_ids in grouped:
#             print(f"{task} | Saline IDs ({section_name}): {len(saline_ids)}")
#             print(f"{task} | Ghrelin IDs ({section_name}): {len(ghrelin_ids)}")

#             if not saline_ids or not ghrelin_ids:
#                 print(f"Skipping {task}: one or both groups are empty.")
#                 continue

#             _, excel_path, fig_path = run_curvature_analysis_groups(
#                 id_lists=[saline_ids, ghrelin_ids],
#                 labels=["Saline", "Ghrelin"],
#                 analysis_name=f"paper2025_{section_name}_{task}",
#                 bodypart="Midback",
#                 how="mean",
#                 smoothing_window=5,
#                 speed_thresh=0.01,
#                 likelihood_threshold=0.5,
#                 normalization=False,
#                 plot_type="bar",
#             )
#             excel_paths.append(excel_path)
#             print(f"Saved {task} curvature Excel: {excel_path}")
#             print(f"Saved {task} curvature figure: {fig_path}")

#         print(f"Combining curvature summaries: {section_name}")
#         _combine_task_excels(
#             excel_paths=excel_paths,
#             output_name=f"paper2025_{section_name}_alltasks",
#             feature="curvature",
#         )


# def run_angle_sections() -> None:
#     from scripts.pipelines.run_angle_analysis import run_angle_analysis_groups

#     sections = [
#         ("10x_ibu", 10, False),
#         ("2x_ibu_modulation_null", 2, True),
#     ]

#     for section_name, dose_mult, require_modulation_null in sections:
#         print(f"Running angle analysis: {section_name}")
#         grouped = _fetch_task_groups(
#             dose_mult=dose_mult,
#             require_modulation_null=require_modulation_null,
#         )

#         excel_paths: list[str] = []
#         for task, saline_ids, ghrelin_ids in grouped:
#             print(f"{task} | Saline IDs ({section_name}): {len(saline_ids)}")
#             print(f"{task} | Ghrelin IDs ({section_name}): {len(ghrelin_ids)}")

#             if not saline_ids or not ghrelin_ids:
#                 print(f"Skipping {task}: one or both groups are empty.")
#                 continue

#             _, excel_path, fig_path = run_angle_analysis_groups(
#                 id_lists=[saline_ids, ghrelin_ids],
#                 labels=["Saline", "Ghrelin"],
#                 analysis_name=f"paper2025_{section_name}_{task}",
#                 likelihood_threshold=0.8,
#                 metric="median",
#                 plot_type="bar",
#             )
#             excel_paths.append(excel_path)
#             print(f"Saved {task} angle Excel: {excel_path}")
#             print(f"Saved {task} angle figure: {fig_path}")

#         print(f"Combining angle summaries: {section_name}")
#         _combine_task_excels(
#             excel_paths=excel_paths,
#             output_name=f"paper2025_{section_name}_alltasks",
#             feature="angle",
#         )


def main() -> None:
    # Set paper-specific env before importing project modules that load config.
    os.environ["ENV_FILE"] = ".env.paper2025"
    _force_env_from_file(".env.paper2025")

    run_speed_sections()
    # run_curvature_sections()
    # run_angle_sections()


if __name__ == "__main__":
    main()
