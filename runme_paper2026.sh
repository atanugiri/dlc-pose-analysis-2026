#!/bin/bash
set -euo pipefail

export ENV_FILE=.env.paper2026
export PAPER_TAG=paper2026

RESULTS_ROOT="results/${PAPER_TAG}"

echo "Running speed analysis..."
python -m scripts.pipelines.run_speed_analysis --task ToyRAT --individual m1 --plot-type box
python -m scripts.pipelines.run_speed_analysis --task ToyStick --plot-type box

echo "Combining ToyRAT + ToyStick speed..."
python -m scripts.pipelines.combine_task_analysis \
  "${RESULTS_ROOT}/speed_analysis/toyrat_head_sw_None_lt_None_speed_summary.xlsx" \
  "${RESULTS_ROOT}/speed_analysis/toystick_head_sw_None_lt_None_speed_summary.xlsx" \
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
  "${RESULTS_ROOT}/curvature_analysis/toyrat_mean_midback_sw_5_lt_0.5_st_0.01_curvature_summary.xlsx" \
  "${RESULTS_ROOT}/curvature_analysis/toystick_mean_midback_sw_5_lt_0.5_st_0.01_curvature_summary.xlsx" \
  --feature curvature --output-name toyrat_toystick --plot-type box

# Angle analysis for all tasks
python -m scripts.pipelines.run_angle_analysis \
--task ToyRAT --individual m1 --likelihood-threshold 0.8 --metric median --plot-type box
python -m scripts.pipelines.run_angle_analysis \
--task ToyStick --likelihood-threshold 0.8 --metric median --plot-type box

echo "Combining ToyRAT + ToyStick angle..."
python -m scripts.pipelines.combine_task_analysis \
  "${RESULTS_ROOT}/angle_analysis/toyrat_lt_0.8_angle_summary.xlsx" \
  "${RESULTS_ROOT}/angle_analysis/toystick_lt_0.8_angle_summary.xlsx" \
  --feature angle --output-name toyrat_toystick --plot-type box
