#!/bin/bash
set -euo pipefail

# Save frequently-run analysis commands
# Speed analysis for all tasks
TASKS=('ChickenBroth' 'ChocolateMilk' 'LightOnly' 'FoodOnly' 'FoodLight' 'ToyOnly' 'ToyLight' 'ToyStick')
TASKS=('ChocolateMilk' 'FoodOnly' 'FoodLight' 'ToyStick')

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


echo "Running speed analysis..."
python -m scripts.pipelines.run_speed_analysis --task ToyRAT --individual m1
python -m scripts.pipelines.run_speed_analysis --task ToyStick

echo "Combining ToyRAT + ToyStick speed..."
python -m scripts.pipelines.combine_task_analysis \
  results/speed_analysis/toyrat_head_*.csv \
  results/speed_analysis/toystick_head_*.csv \
  --output-name toyrat_toystick

# Curvature analysis for all tasks
echo "Running curvature analysis..."
python -m scripts.pipelines.run_curvature_analysis \
  --task ToyRAT --individual m1 --likelihood-threshold 0.5 --normalization false

python -m scripts.pipelines.run_curvature_analysis \
  --task ToyStick --likelihood-threshold 0.5 --normalization false

echo "Combining ToyRAT + ToyStick curvature..."
python -m scripts.pipelines.combine_task_analysis \
  results/curvature_analysis/toyrat*.csv \
  results/curvature_analysis/toystick*.csv \
  --feature 'curvature' --output-name toyrat_toystick

# Angle analysis for all tasks
python -m scripts.pipelines.run_angle_analysis --task ToyRAT --individual m1
python -m scripts.pipelines.run_angle_analysis --task ToyStick

echo "Combining ToyRAT + ToyStick angle..."
python -m scripts.pipelines.combine_task_analysis \
  results/angle_analysis/toyrat*.csv \
  results/angle_analysis/toystick*.csv \
  --feature 'angle' --output-name toyrat_toystick