# DLC Pose Analysis Pipeline

## Step 1: Extract filenames
Run:
./scripts/extract_filenames.sh

## Step 2: Import into PostgreSQL
\copy raw_file_list(file_name) FROM 'data/raw/raw_black_chickenbroth_filenames.txt'

## Step 3: Parse metadata
Run 02_views.sql

## Step 4: Insert into experimental_metadata
Run 03_import.sql