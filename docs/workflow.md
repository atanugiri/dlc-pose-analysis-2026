# DLC Pose Analysis Pipeline

## Step 1: Extract filenames
Run:
./scripts/extract_filenames.sh

## Step 2: Import into PostgreSQL
\copy raw_file_list(file_name) FROM 'data/raw/raw_black_chickenbroth_filenames.txt'

## Step 3: Parse metadata
Run 02_views.sql

## Step 4: Insert into experimental_metadata
Run 03_import.sql

---

## Step 5: Keypoint-MoSeq syllable analysis

Keypoint-MoSeq (KPMS) segments pose trajectories into discrete, reusable
behavioural motifs called *syllables* and produces a frame-by-frame time-series
that can be aligned with the original video.

### Setup

```bash
conda env create -f environment.yml
conda activate kpms
```

### Interactive notebook (recommended)

Open the notebook and follow the cells in order:

```bash
jupyter lab notebooks/keypoint_moseq_pipeline.ipynb
```

Key variables to set in cell 1:

| Variable | Description |
|----------|-------------|
| `DLC_PROJECT_ROOT` | Path to your DLC project (e.g. `~/Downloads/dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04`) |
| `POSE_DIR` | Subdirectory containing filtered DLC CSV files (default: `<root>/filtered_pose_data`) |
| `VIDEO_DIR` | Subdirectory containing video files (default: `<root>/videos`) |
| `FPS` | Recording frame rate (default: 30) |

### Command-line script

```bash
conda activate kpms

python scripts/kpms_pipeline.py \
  --pose-dir ~/Downloads/dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04/filtered_pose_data \
  --project-dir kpms_project \
  --num-ar-iters 50 \
  --num-iters 200 \
  --results-dir results
```

Run `python scripts/kpms_pipeline.py --help` for all options.

### Outputs

| File | Description |
|------|-------------|
| `kpms_project/config.yml` | KPMS configuration (edit before re-running) |
| `kpms_project/pca.p` | Fitted PCA |
| `kpms_project/<model_name>/` | Model checkpoints |
| `results/syllable_timeseries.csv` | Frame-by-frame syllable labels + timestamps |
| `results/syllable_timeseries.png` | Visualisation of syllable sequences |
| `results/syllable_clips/` | (optional) Representative video clips per syllable |

### Aligning the syllable time-series with video

`syllable_timeseries.csv` contains columns `recording`, `frame`, `syllable`,
and `time_sec`.  To jump to a specific syllable in a video player, filter rows
for the desired syllable and note the `time_sec` values, then seek to that
timestamp.  The `--export-clips` flag (script) or `EXPORT_CLIPS = True`
(notebook) will generate short MP4 clips representative of each syllable
automatically.