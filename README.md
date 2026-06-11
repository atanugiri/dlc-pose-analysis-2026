# DLC Pose Analysis 2026

This repository contains DeepLabCut-based trajectory analysis code for saline vs ghrelin comparisons.

## Repository Layout

- `data/filtered_pose_data/`: filtered DeepLabCut `.h5` files
- `database/`: schema, views, and import SQL
- `scripts/features/`: feature extraction code
- `scripts/plots/`: plotting utilities (including `group_comparison_plot`)
- `scripts/utils/`: utility scripts (including the task summary combiner)
- `results/`: generated Excel summaries and figures
- `notebooks/`: exploratory analysis notebooks

## Environment

Use the conda environment used for this project (example):

```bash
conda env create -f environment.yml
conda activate ghrelin
```

Do not commit real `.env` files. For reproducibility, keep secrets local and share a sanitized template.

Create a local env file (`.env` or `.env.paper2026`) in project root with your PostgreSQL settings, for example:

```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=atanugiri
DB_PASSWORD=
DB_NAME=dlc_pose_analysis_2026
MAZE_SIZE_CM=64
```

`MAZE_SIZE_CM=64` is required for reproducibility because normalized coordinates are converted to physical units (cm) using this value.

Recommended reproducible workflow:

1. Add an `.env.example` file to the repo with placeholder (non-secret) values.
2. Each user copies it locally, for example:

```bash
cp .env.example .env.paper2026
```

3. Fill in local credentials in `.env.paper2026`.
4. Run with that env file selected:

```bash
ENV_FILE=.env.paper2026 python runme_paper2026.py
```

This repo already ignores `.env` and `.env.*`, so secrets stay local.

Then run analyses from repository root:

```bash
python runme_paper2026.py
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
python -m scripts.db.import_project_csvs_to_postgres --csv-file data/experimental_metadata.csv --table experimental_metadata
python -m scripts.db.import_project_csvs_to_postgres --csv-file data/maze_map.csv --table maze_map
```

Notes:

- The import script uses pandas `to_sql` with fixed `if_exists=replace` behavior.
- Preferred setup: configure local DB settings in `.env`/`.env.paper2026` (loaded by `python-dotenv` in `scripts/config.py`).
- Fallback setup: edit defaults in `scripts/config.py` (`DB_CONNECT_KWARGS`).
- Run the import command above for each CSV file you want to load.
- Run analyses normally.
- For reproducibility, include the two CSV files plus this import command in your submission instructions.

## Analysis Run

Use [runme_paper2026.sh](runme_paper2026.sh) as the source of truth for exact execution steps.

```bash
bash runme_paper2026.sh
```

For custom runs, use the main analysis entrypoint.

```bash
python runme_paper2026.py
```

## Statistical Tests

Analyses that call `group_comparison_plot` support:

- `welch` (default, two-tailed Welch t-test)
- `mann_whitney`

Pass with `--test` in your configured run entrypoint as needed.

```bash
python runme_paper2026.py
```

## Outputs

- Speed outputs: `results/speed_analysis/`
- Curvature outputs: `results/curvature_analysis/`
- Angle outputs: `results/angle_analysis/`

Each run writes:

1. Summary Excel file (`*_summary.xlsx`)
2. Plot (`*_barplot.pdf` or `*_boxplot.pdf`)