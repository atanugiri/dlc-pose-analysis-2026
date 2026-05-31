# DLC Pose Analysis 2026

This repository contains DeepLabCut-based trajectory analysis pipelines for saline vs ghrelin comparisons.

## Repository Layout

- `data/filtered_pose_data/`: filtered DeepLabCut `.h5` files
- `database/`: schema, views, and import SQL
- `scripts/features/`: feature extraction code
- `scripts/plots/`: plotting utilities (including `group_comparison_plot`)
- `scripts/pipelines/`: runnable analysis pipelines
- `results/`: generated Excel summaries and figures
- `notebooks/`: exploratory analysis notebooks

## Environment

Use the conda environment used for this project (example):

```bash
conda activate ghrelin
```

Then run pipelines with module syntax from repository root:

```bash
python -m scripts.pipelines.run_speed_analysis --task ToyRAT
```

## CSV Export And Rebuild Workflow

For code submission, the PostgreSQL-backed metadata tables can be exported as CSV and later re-imported.

Export current DB tables to CSV files in `data/` with `psql`:

```bash
psql -d dlc_pose_analysis_2026 -c "\copy (SELECT * FROM public.experimental_metadata ORDER BY id) TO 'data/experimental_metadata.csv' CSV HEADER"
psql -d dlc_pose_analysis_2026 -c "\copy (SELECT * FROM public.maze_map ORDER BY task, genotype, animal_name, start_date, end_date) TO 'data/maze_map.csv' CSV HEADER"
```

This writes:

- `data/experimental_metadata.csv`
- `data/maze_map.csv`

Recreate DB table contents from those CSV files:

```bash
python -m scripts.db.import_project_csvs_to_postgres
```

Notes:

- The import script uses pandas `to_sql` with fixed `if_exists=replace` behavior.
- Reproducibility without `.env`: open `scripts/config.py`.
- Edit defaults in `DB_CONNECT_KWARGS` to match your local PostgreSQL (`host`, `port`, `user`, `password`, `database`).
- Run `python -m scripts.db.import_project_csvs_to_postgres`.
- Run analysis pipelines normally.
- For reproducibility, include the two CSV files plus this import command in your submission instructions.

## Analysis Pipelines

Use [runme.sh](runme.sh) as the source of truth for exact execution steps.

```bash
bash runme.sh
```

For custom runs, use each pipeline module with `--help`.

## Statistical Tests

Pipelines that call `group_comparison_plot` support:

- `welch` (default, two-tailed Welch t-test)
- `mann_whitney`

Pass with `--test`, for example:

```bash
python -m scripts.pipelines.run_speed_analysis --task ToyRAT --test mann_whitney
```

## Outputs

- Speed outputs: `results/speed_analysis/`
- Curvature outputs: `results/curvature_analysis/`
- Angle outputs: `results/angle_analysis/`

Each run writes:

1. Summary Excel file (`*_summary.xlsx`)
2. Plot (`*_barplot.pdf` or `*_boxplot.pdf`)