#!/bin/bash
set -euo pipefail

INPUT_DIR="../dlc-pose-estimation/Black-ChickenBroth/raw_pose_data"
OUTPUT_FILE="data/staging/raw_black_chickenbroth_filenames.txt"

mkdir -p data/staging

[[ -d "$INPUT_DIR" ]] || { echo "Input directory not found: $INPUT_DIR"; exit 1; }

find "$INPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -name "*.h5" \
  -exec basename {} \; \
  | sort > "$OUTPUT_FILE"

echo "Saved filenames to $OUTPUT_FILE"