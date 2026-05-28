# DLC Pose Analysis 2026

This repository contains DeepLabCut-based trajectory analysis pipelines for saline vs ghrelin comparisons.

## Repository Layout

- `data/filtered_pose_data/`: filtered DeepLabCut `.h5` files
- `data/raw_pose_data/`: raw DeepLabCut outputs
- `database/`: schema, views, and import SQL
- `scripts/features/`: feature extraction code
- `scripts/plots/`: plotting utilities (including `barplot_mean_se`)
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
python -m scripts.pipelines.run_speed_analysis --task ChickenBroth
```

## Database Setup (PostgreSQL)

1. Start PostgreSQL.
2. Create database:

```sql
CREATE DATABASE dlc_pose_analysis_2026;
```

3. Run SQL files in order:

- `database/01_schema.sql`
- `database/02_views.sql`
- `database/03_import.sql`
- `database/04_maze_map.sql`
- `database/04_set_maze_number_from_maze_map.sql`

## Analysis Pipelines

All pipeline scripts compare saline (`Y`) vs ghrelin (`P`) groups and save both Excel summaries and barplots.

### 1) Speed

```bash
python -m scripts.pipelines.run_speed_analysis \
  --task ChickenBroth \
  --bodypart Head \
  --how mean
```

### 2) Curvature

```bash
python -m scripts.pipelines.run_curvature_analysis \
  --task ChickenBroth \
  --bodypart Midback \
  --how mean \
  --smoothing-window 5 \
  --speed-thresh 0.01
```

### 3) Head-Body Misalignment Angle

```bash
python -m scripts.pipelines.run_angle_analysis \
  --task ChickenBroth \
  --metric p95
```

### 4) Combine Multiple Task Summaries

```bash
python -m scripts.pipelines.combine_task_analysis \
  results/speed_analysis/task_a_speed_summary.xlsx \
  results/speed_analysis/task_b_speed_summary.xlsx \
  --feature speed \
  --output-name combined_tasks
```

## Statistical Tests

Pipelines that call `barplot_mean_se` support:

- `welch` (default, two-tailed Welch t-test)
- `welch_greater`
- `welch_less`
- `mann_whitney`

Pass with `--test`, for example:

```bash
python -m scripts.pipelines.run_speed_analysis --task ChickenBroth --test mann_whitney
```

## Outputs

- Speed outputs: `results/speed_analysis/`
- Curvature outputs: `results/curvature_analysis/`
- Angle outputs: `results/angle_analysis/`

Each run writes:

1. Summary Excel file (`*_summary.xlsx`)
2. Bar plot (`*_barplot.pdf`)