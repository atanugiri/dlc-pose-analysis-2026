INSERT INTO experimental_metadata (
    video_name,
    genotype,
    task,
    session_date,
    animal_name,
    treatment,
    trial
)
SELECT
    video_name,
    genotype,
    task,
    session_date,
    animal_name,
    treatment,
    trial
FROM parsed_black_chickenbroth_files;