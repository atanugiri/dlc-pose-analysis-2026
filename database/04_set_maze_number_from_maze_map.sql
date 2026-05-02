-- 04_set_maze_number_from_maze_map.sql
-- Populate experimental_metadata.maze_number from maze_map.
-- Run the preview query first to verify rows, then run the update inside a transaction.

BEGIN;

-- Preview how many rows will be updated.
-- Run this and inspect the count before proceeding.
SELECT COUNT(*) AS rows_to_update
FROM experimental_metadata e
JOIN maze_map m
  ON e.task = m.task
  AND e.genotype = m.genotype
  AND e.animal_name = m.animal_name
WHERE e.maze_number IS NULL
  AND e.session_date BETWEEN m.start_date AND m.end_date;

-- If the preview looks correct, run the UPDATE below.
UPDATE experimental_metadata e
SET maze_number = m.maze_number
FROM maze_map m
WHERE e.task = m.task
  AND e.genotype = m.genotype
  AND e.animal_name = m.animal_name
  AND e.session_date BETWEEN m.start_date AND m.end_date
  AND e.maze_number IS NULL;

COMMIT;

-- Notes:
-- - This script is idempotent for rows where maze_number is NULL.
-- - To test without persisting changes, run the preview SELECT and/or
--   replace the UPDATE with a SELECT of the expected output columns.
