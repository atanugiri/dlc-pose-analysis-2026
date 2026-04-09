-- Create table
CREATE TABLE experimental_metadata (
    id              INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_name      TEXT NOT NULL UNIQUE,
    genotype        TEXT,
    task            TEXT NOT NULL,
    session_date    DATE,
    animal_name     TEXT,
    treatment       TEXT NOT NULL,
    trial           INTEGER,
    file_path       TEXT
);

CREATE TABLE raw_file_list (
    id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    file_name  TEXT NOT NULL UNIQUE
);