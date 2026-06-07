from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values


def _force_env_from_file(env_file: str) -> None:
    """Load env file values and force them into process env."""
    env_path = Path(__file__).resolve().parent / env_file
    file_vars = dotenv_values(env_path)
    for key, value in file_vars.items():
        if value is None:
            continue
        os.environ[key] = value


def main() -> None:
    # Set paper-specific env before importing project modules that load config.
    os.environ["ENV_FILE"] = ".env.paper2025"
    _force_env_from_file(".env.paper2025")

    from scripts.db.get_filtered_ids import get_filtered_ids
    from scripts.pipelines.run_speed_analysis import run_speed_analysis_groups

    tasks = ["FoodOnly", "ToyOnly", "LightOnly", "FoodLight", "ToyLight"]
    print("Running paper2025 speed analysis for 10x IBU task-by-task...")

    for task in tasks:
        saline_ids = get_filtered_ids({"task": task, "treatment": "Y", "dose_mult": 10})
        ghrelin_ids = get_filtered_ids({"task": task, "treatment": "P", "dose_mult": 10})

        print(f"{task} | Saline IDs (10x IBU): {len(saline_ids)}")
        print(f"{task} | Ghrelin IDs (10x IBU): {len(ghrelin_ids)}")

        if not saline_ids or not ghrelin_ids:
            print(f"Skipping {task}: one or both groups are empty.")
            continue

        _, excel_path, fig_path = run_speed_analysis_groups(
            id_lists=[saline_ids, ghrelin_ids],
            labels=["Saline", "Ghrelin"],
            analysis_name=f"paper2025_10x_ibu_{task}",
            bodypart="Head",
            plot_type="bar",
        )

        print(f"Saved {task} speed Excel: {excel_path}")
        print(f"Saved {task} speed figure: {fig_path}")


if __name__ == "__main__":
    main()
