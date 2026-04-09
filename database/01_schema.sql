-- Create table
DROP TABLE experimental_metadata;

CREATE TABLE experimental_metadata (
    id                   INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_name           TEXT NOT NULL UNIQUE,
    genotype             TEXT,
    task                 TEXT NOT NULL,
    session_date         DATE,
    animal_name          TEXT,
    treatment            TEXT NOT NULL,
    trial                INTEGER,
    raw_pose_file        TEXT UNIQUE,
    filtered_pose_file   TEXT UNIQUE,
    multi_animal         BOOLEAN NOT NULL DEFAULT FALSE;
);

CREATE TABLE raw_file_list (
    id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_name  TEXT NOT NULL UNIQUE
);

