#!/bin/bash
set -euo pipefail

# Save frequently-run analysis commands

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

echo "Running curvature analysis..."
python -m scripts.pipelines.run_curvature_analysis --task ChickenBroth
python -m scripts.pipelines.run_curvature_analysis --task ToyLight

echo "All analyses complete!"