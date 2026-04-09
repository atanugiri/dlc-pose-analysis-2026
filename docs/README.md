# DLC Pose Analysis 2026

This repository contains all data and code used for trajectory analysis.

## Structure

- data/filtered_pose_data: DeepLabCut output (.h5 files)
- database/: SQL schema, views, and import scripts
- scripts/: bash scripts for preprocessing
- notebooks/: analysis notebooks
- results/: generated figures and outputs

## Setup

1. Start PostgreSQL
2. Create database:
   CREATE DATABASE dlc_pose_analysis_2026;

3. Run:
   database/01_schema.sql
   database/02_views.sql

4. Extract filenames:
   ./scripts/extract_filenames.sh

5. Import:
   \copy raw_file_list(file_name) FROM 'data/staging/raw_black_chickenbroth_filenames.txt'

6. Insert:
   database/03_import.sql

## Notes
- All filenames are assumed to follow standardized naming format