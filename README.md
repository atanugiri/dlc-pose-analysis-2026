# DLC Pose Analysis 2026

This repository contains all data and code used for trajectory analysis,
including a keypoint-moseq syllable pipeline for the
**ElevatedMazeFood-Atanu-2026-04-04** experiment.

## Structure

- `data/`: staging SQL and file lists
- `database/`: SQL schema, views, and import scripts
- `scripts/`: bash and Python scripts for preprocessing and KPMS pipeline
- `notebooks/`: Jupyter analysis notebooks
- `results/`: generated figures and outputs
- `environment.yml`: conda environment for keypoint-moseq

## Setup

### PostgreSQL metadata pipeline

1. Start PostgreSQL
2. Create database:
   ```sql
   CREATE DATABASE dlc_pose_analysis_2026;
   ```
3. Run:
   ```
   database/01_schema.sql
   database/02_views.sql
   ```
4. Extract filenames:
   ```bash
   ./scripts/extract_filenames.sh
   ```
5. Import:
   ```sql
   \copy raw_file_list(file_name) FROM 'data/staging/raw_black_chickenbroth_filenames.txt'
   ```
6. Insert:
   ```
   database/03_import.sql
   ```

### Keypoint-MoSeq syllable pipeline

Requires filtered DLC CSV files in `filtered_pose_data/` inside the DLC
project directory (`~/Downloads/dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04`).

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate kpms

# Interactive notebook (recommended)
jupyter lab notebooks/keypoint_moseq_pipeline.ipynb

# Or run the command-line script
python scripts/kpms_pipeline.py \
  --pose-dir ~/Downloads/dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04/filtered_pose_data \
  --project-dir kpms_project \
  --num-iters 200 \
  --results-dir results
```

Key outputs:
- `results/syllable_timeseries.csv` – frame-by-frame syllable labels + timestamps
- `results/syllable_timeseries.png` – visualisation of syllable sequences
- `results/syllable_clips/` – (optional) representative video clips per syllable

See [docs/workflow.md](docs/workflow.md) for the full pipeline description.

## Notes
- All filenames are assumed to follow standardized naming format
- Large files (`.h5`, `.csv`, videos) are excluded from version control via `.gitignore`