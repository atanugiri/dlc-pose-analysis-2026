#!/bin/bash
set -euo pipefail

export ENV_FILE=.env.paper2025
export PAPER_TAG=paper2025

RESULTS_ROOT="results/${PAPER_TAG}"

GROUP_SPECS=(
  "WT_Saline|WT|Y|1.0|None"
  "WT_Ghrelin|WT|P|1.0|None"
)

SPEED_ARGS=()
for spec in "${GROUP_SPECS[@]}"; do
  IFS='|' read -r label genotype treatment dose_mult modulation <<< "${spec}"

  ids_csv="$(python - "${genotype}" "${treatment}" "${dose_mult}" "${modulation}" <<'PY'
import sys
from scripts.db.db_utils import get_ids_by_filters

genotype, treatment, dose_mult, modulation = sys.argv[1:5]

genotype = None if genotype == "None" else genotype
treatment = None if treatment == "None" else treatment
dose_mult = None if dose_mult == "None" else float(dose_mult)
modulation = None if modulation == "None" else modulation

ids = get_ids_by_filters(
    genotype=genotype,
    treatment=treatment,
    dose_mult=dose_mult,
    modulation=modulation,
)

print(",".join(str(record_id) for record_id in ids))
PY
)"

  if [[ -z "${ids_csv}" ]]; then
    echo "No IDs found for group ${label}; skipping this group."
    continue
  fi

  SPEED_ARGS+=(--id-list "${ids_csv}" --label "${label}")
done

if (( ${#SPEED_ARGS[@]} < 4 )); then
  echo "Need at least two non-empty groups in GROUP_SPECS for speed analysis."
  exit 1
fi

python -m scripts.pipelines.run_speed_analysis "${SPEED_ARGS[@]}" --plot-type box

python -m scripts.pipelines.run_curvature_analysis \
  --task ToyRAT --individual m1 --likelihood-threshold 0.5 --normalization false --plot-type box

python -m scripts.pipelines.run_curvature_analysis \
  --task ToyStick --likelihood-threshold 0.5 --normalization false --plot-type box

python -m scripts.pipelines.combine_task_analysis \
  "${RESULTS_ROOT}/curvature_analysis/toyrat_mean_midback_sw_5_lt_0.5_st_0.01_curvature_summary.xlsx" \
  "${RESULTS_ROOT}/curvature_analysis/toystick_mean_midback_sw_5_lt_0.5_st_0.01_curvature_summary.xlsx" \
  --feature curvature --output-name toyrat_toystick --plot-type box

python -m scripts.pipelines.run_angle_analysis \
  --task ToyRAT --individual m1 --likelihood-threshold 0.8 --metric median --plot-type box

python -m scripts.pipelines.run_angle_analysis \
  --task ToyStick --likelihood-threshold 0.8 --metric median --plot-type box

python -m scripts.pipelines.combine_task_analysis \
  "${RESULTS_ROOT}/angle_analysis/toyrat_lt_0.8_angle_summary.xlsx" \
  "${RESULTS_ROOT}/angle_analysis/toystick_lt_0.8_angle_summary.xlsx" \
  --feature angle --output-name toyrat_toystick --plot-type box
