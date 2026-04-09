-- Create table
CREATE TABLE experimental_metadata (
    id              integer generated always as identity primary key,
    video_name      text not null unique,
    genotype        text,
    task            text not null,
    session_date    date,
    animal_name     text,
    treatment       text not null,
    trial           integer,
    file_path       text
);