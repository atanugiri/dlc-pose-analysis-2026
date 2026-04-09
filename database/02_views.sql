CREATE OR REPLACE VIEW parsed_black_chickenbroth_files AS
SELECT
    file_name AS video_name,
    split_part(file_name, '_', 1) AS task,
    TO_DATE(
        split_part(file_name, '_', 2) || '_' ||
        split_part(file_name, '_', 3) || '_' ||
        split_part(file_name, '_', 4),
        'MM_DD_YY'
    ) AS session_date,
    split_part(file_name, '_', 5) AS genotype,
    split_part(file_name, '_', 6) AS animal_name,
    regexp_replace(
        split_part(file_name, '_', 7),
        '^Trial([0-9]+).*',
        '\1'
    )::INTEGER AS trial,
    'unknown'::TEXT AS treatment
FROM raw_file_list;