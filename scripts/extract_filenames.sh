#!/bin/bash

INPUT_DIR="../dlc-pose-estimation/Black-ChickenBroth/filtered_pose_data"
OUTPUT_FILE="data/raw/raw_black_chickenbroth_filenames.txt"

mkdir -p data/raw

find "$INPUT_DIR" \
  -maxdepth 1 \
  -type f \
  -name "*.h5" \
  -exec basename {} \; \
  | sort > "$OUTPUT_FILE"

echo "Saved filenames to $OUTPUT_FILE"