#!/bin/bash
set -euo pipefail

echo "Running speed analysis..."
python -m scripts.pipelines.run_speed_analysis --task ToyRAT --individual m1 --plot-type box
python -m scripts.pipelines.run_speed_analysis --task ToyStick --plot-type box

echo "Combining ToyRAT + ToyStick speed..."
python -m scripts.pipelines.combine_task_analysis \
  results/speed_analysis/toyrat_head_sw_None_lt_None_speed_summary.xlsx \
  results/speed_analysis/toystick_head_sw_None_lt_None_speed_summary.xlsx \
  --output-name toyrat_toystick \
  --feature speed --plot-type box

# Curvature analysis for all tasks
echo "Running curvature analysis..."
python -m scripts.pipelines.run_curvature_analysis \
  --task ToyRAT --individual m1 --likelihood-threshold 0.5 --normalization false --plot-type box

python -m scripts.pipelines.run_curvature_analysis \
  --task ToyStick --likelihood-threshold 0.5 --normalization false --plot-type box

echo "Combining ToyRAT + ToyStick curvature..."
python -m scripts.pipelines.combine_task_analysis \
  results/curvature_analysis/toyrat_mean_midback_sw_5_lt_0.5_st_0.01_curvature_summary.xlsx \
  results/curvature_analysis/toystick_mean_midback_sw_5_lt_0.5_st_0.01_curvature_summary.xlsx \
  --feature curvature --output-name toyrat_toystick --plot-type box

# Angle analysis for all tasks
python -m scripts.pipelines.run_angle_analysis \
--task ToyRAT --individual m1 --likelihood-threshold 0.8 --metric median --plot-type box
python -m scripts.pipelines.run_angle_analysis \
--task ToyStick --likelihood-threshold 0.8 --metric median --plot-type box

echo "Combining ToyRAT + ToyStick angle..."
python -m scripts.pipelines.combine_task_analysis \
  results/angle_analysis/toyrat_lt_0.8_median_summary.xlsx \
  results/angle_analysis/toystick_lt_0.8_median_summary.xlsx \
  --feature median --output-name toyrat_toystick --plot-type box
