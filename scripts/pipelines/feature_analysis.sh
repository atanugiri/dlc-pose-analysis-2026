#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=/work/agiri/logs/%x-%j.out
#SBATCH --error=/work/agiri/logs/%x-%j.err
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=32G

set -euo pipefail

# Create necessary directories
mkdir -p "$WORK/logs"
mkdir -p "$WORK/.cache"
mkdir -p "$WORK/pip-cache"
mkdir -p "$WORK/mplconfig"
mkdir -p "$WORK/tmp"

# Set up environment variables
export CONDA_ENV="$HOME/miniconda3/envs/ghrelin"
export PIP_CACHE_DIR="$WORK/pip-cache"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/mplconfig"
export TMPDIR="$WORK/tmp"
export TMP="$WORK/tmp"
export TEMP="$WORK/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Activate conda environment
echo "Job started on $(hostname) at $(date)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Navigate to project directory
cd "$WORK/dlc-pose-analysis-2026" || exit 1

# Define tasks to iterate over
TASKS=('LightOnly' 'ChickenBroth' 'ToyLight' 'FoodOnly' 'FoodLight' 'ChocolateMilk' 'ToyOnly' 'ToyStick')

# Parse command line arguments (default values)
echo "Running speed analysis for all tasks:"
echo "  Tasks: ${TASKS[@]}"
echo ""

# Iterate over all tasks
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"
    python -m scripts.pipelines.run_speed_analysis \
        --task "$TASK"
    echo "  ✓ Completed $TASK"
    echo ""
done

echo "All tasks completed"
