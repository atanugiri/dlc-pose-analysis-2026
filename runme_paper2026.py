from __future__ import annotations

import os


def main() -> None:
    # Set paper-specific env before importing project modules that load config.
    os.environ["ENV_FILE"] = ".env.paper2026"

    from scripts.db.get_filtered_ids import get_filtered_ids
    from scripts.pipelines.run_speed_analysis import run_speed_analysis_groups

    print("Running speed analysis (paper2026) with dynamic DB filters...")

    toy_rat_saline = get_filtered_ids({"task": "ToyRAT", "treatment": "Y"})
    toy_rat_ghrelin = get_filtered_ids({"task": "ToyRAT", "treatment": "P"})
    toy_stick_saline = get_filtered_ids({"task": "ToyStick", "treatment": "Y"})
    toy_stick_ghrelin = get_filtered_ids({"task": "ToyStick", "treatment": "P"})

    _, excel_path_rat, fig_path_rat = run_speed_analysis_groups(
        id_lists=[toy_rat_saline, toy_rat_ghrelin],
        labels=["Saline", "Ghrelin"],
        analysis_name="toyrat",
        bodypart="Head",
        individual="m1",
        plot_type="box",
    )

    _, excel_path_stick, fig_path_stick = run_speed_analysis_groups(
        id_lists=[toy_stick_saline, toy_stick_ghrelin],
        labels=["Saline", "Ghrelin"],
        analysis_name="toystick",
        bodypart="Head",
        plot_type="box",
    )

    print(f"Saved ToyRAT Excel: {excel_path_rat}")
    print(f"Saved ToyRAT figure: {fig_path_rat}")
    print(f"Saved ToyStick Excel: {excel_path_stick}")
    print(f"Saved ToyStick figure: {fig_path_stick}")


if __name__ == "__main__":
    main()
