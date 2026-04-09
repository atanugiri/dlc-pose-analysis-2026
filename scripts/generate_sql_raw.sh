#!/bin/bash
set -euo pipefail

INPUT_FILE="data/staging/raw_black_chickenbroth_filenames.txt"
OUTPUT_SQL="data/staging/insert_raw_metadata.sql"

> "$OUTPUT_SQL"

while IFS= read -r filename;  do
    base="${filename%.h5}"

    IFS='_' read -r task month day year session_token animal trial_rest <<< "$base"

    trial=$(echo "$trial_rest" | sed -E 's/Trial([0-9]+).*/\1/')
    session_date="20${year}-${month}-${day}"
    treatment=$(echo "$session_token" | sed -E 's/^S[0-9]+([YP])$/\1/')
    raw_pose_path="$filename"
    video_name=$(echo "$base" | sed -E 's/(Trial[0-9]+).*/\1/')

    echo "INSERT INTO experimental_metadata (video_name, task, session_date, animal_name, treatment, trial, raw_pose_path)
VALUES ('$video_name', '$task', '$session_date', '$animal', '$treatment', $trial, '$raw_pose_path');" >> "$OUTPUT_SQL"

done < "$INPUT_FILE"

echo "SQL file generated at $OUTPUT_SQL"