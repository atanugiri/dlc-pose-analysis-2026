#!/bin/bash
set -euo pipefail

INPUT_FILE="data/staging/raw_ElevatedMazeFood-Atanu-2026-04-04_filenames.txt"
OUTPUT_SQL="data/staging/insert_raw_metadata.sql"

> "$OUTPUT_SQL"

sql_escape() {
    printf "%s" "$1" | sed "s/'/''/g"
}

while IFS= read -r filename; do
    [[ -z "$filename" ]] && continue

    base="${filename%.h5}"
    IFS='_' read -r -a parts <<< "$base"

    if (( ${#parts[@]} < 5 )); then
        echo "Skipping malformed filename: $filename" >&2
        continue
    fi

    task="${parts[0]}"
    month="${parts[1]}"
    day="${parts[2]}"
    year="${parts[3]}"
    session_date=$(printf "%04d-%02d-%02d" $((10#$year + 2000)) $((10#$month)) $((10#$day)))
    session_idx=-1
    treatment=""
    for ((i=4; i<${#parts[@]}; i++)); do
        token="${parts[i]}"
        if [[ "$token" =~ [YP]$ ]]; then
            session_idx=$i
            treatment="${token: -1}"
            break
        fi
    done

    if (( session_idx == -1 )); then
        echo "Skipping, could not detect treatment token: $filename" >&2
        continue
    fi

    trial="NULL"
    animal_parts=()

    for ((i=session_idx+1; i<${#parts[@]}; i++)); do
        token="${parts[i]}"

        # Trial token
        if [[ "$token" =~ ^Trial([0-9]+)DLC$ ]]; then
            trial="${BASH_REMATCH[1]}"
            break
        elif [[ "$token" =~ ^Trial([0-9]+)$ ]]; then
            trial="${BASH_REMATCH[1]}"
            break
        fi

        # Token with animal name glued to DLC, like PinkyDLC
        if [[ "$token" == *DLC* ]]; then
            animal_piece="${token%%DLC*}"
            [[ -n "$animal_piece" ]] && animal_parts+=("$animal_piece")
            break
        fi

        # Extra safety
        if [[ "$token" == *Resnet* ]]; then
            break
        fi

        animal_parts+=("$token")
    done

    if (( ${#animal_parts[@]} == 0 )); then
        animal_name=""
    else
        animal_name=$(IFS=_; echo "${animal_parts[*]}")
    fi

    # video_name = filename before DLC suffix
    video_name=$(echo "$base" | sed -E 's/DLC.*$//')

    task_esc=$(sql_escape "$task")
    animal_esc=$(sql_escape "$animal_name")
    raw_pose_file_esc=$(sql_escape "$filename")
    video_name_esc=$(sql_escape "$video_name")

    if [[ "$trial" == "NULL" ]]; then
        cat >> "$OUTPUT_SQL" <<EOF
INSERT INTO experimental_metadata
(video_name, task, session_date, animal_name, treatment, trial, raw_pose_file)
VALUES
('$video_name_esc', '$task_esc', '$session_date', '$animal_esc', '$treatment', NULL, '$raw_pose_file_esc');
EOF
    else
        cat >> "$OUTPUT_SQL" <<EOF
INSERT INTO experimental_metadata
(video_name, task, session_date, animal_name, treatment, trial, raw_pose_file)
VALUES
('$video_name_esc', '$task_esc', '$session_date', '$animal_esc', '$treatment', $trial, '$raw_pose_file_esc');
EOF
    fi

done < "$INPUT_FILE"

echo "SQL file generated at $OUTPUT_SQL"