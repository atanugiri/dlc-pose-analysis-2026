DROP TABLE IF EXISTS maze_map;

CREATE TABLE maze_map (
    task TEXT NOT NULL,
    genotype TEXT NOT NULL,
    animal_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    maze_number INT NOT NULL,
    PRIMARY KEY (task, genotype, animal_name, start_date, end_date),
    CHECK (end_date >= start_date)      
);

-- Insert data for ChickenBroth task
INSERT INTO maze_map (task, genotype, animal_name, start_date, end_date, maze_number) 
VALUES 
('ChickenBroth', 'black', 'Teddy', '2024-07-01', '2024-07-30', 1),
('ChickenBroth', 'black', 'Joey', '2024-07-01', '2024-07-30', 2),
('ChickenBroth', 'black', 'Loki', '2024-07-01', '2024-07-30', 3),
('ChickenBroth', 'black', 'Hermes', '2024-07-01', '2024-07-30', 4),
('ChickenBroth', 'black', 'Teal', '2024-07-01', '2024-07-30', 1),
('ChickenBroth', 'black', 'Orange', '2024-07-01', '2024-07-30', 2),
('ChickenBroth', 'black', 'Cyan', '2024-07-01', '2024-07-30', 3),
('ChickenBroth', 'black', 'Azul', '2024-07-01', '2024-07-30', 1),
('ChickenBroth', 'black', 'Navy', '2024-07-01', '2024-07-30', 2),
('ChickenBroth', 'black', 'Tan', '2024-07-01', '2024-07-30', 3),
('ChickenBroth', 'black', 'Phoebe', '2024-07-01', '2024-07-30', 1),
('ChickenBroth', 'black', 'Indigo', '2024-07-01', '2024-07-30', 2),
('ChickenBroth', 'black', 'Violet', '2024-07-01', '2024-07-30', 3),
('ChickenBroth', 'black', 'Lilac', '2024-07-01', '2024-07-30', 4),
('ChickenBroth', 'black', 'Bastet', '2024-07-01', '2024-07-30', 1),
('ChickenBroth', 'black', 'Sage', '2024-07-01', '2024-07-30', 2),
('ChickenBroth', 'black', 'Pinky', '2024-07-01', '2024-07-30', 3),
('ChickenBroth', 'black', 'Sky', '2024-07-01', '2024-07-30', 4);

-- Insert data for ChocolateMilk task
INSERT INTO maze_map (task, genotype, animal_name, start_date, end_date, maze_number) 
VALUES 
('ChocolateMilk', 'black', 'Teddy', '2024-07-01', '2024-07-04', 1),
('ChocolateMilk', 'black', 'Joey', '2024-07-01', '2024-07-04', 2),
('ChocolateMilk', 'black', 'Loki', '2024-07-01', '2024-07-04', 3),
('ChocolateMilk', 'black', 'Hermes', '2024-07-01', '2024-07-04', 4),
('ChocolateMilk', 'black', 'Teal', '2024-07-01', '2024-07-04', 1),
('ChocolateMilk', 'black', 'Orange', '2024-07-01', '2024-07-04', 2),
('ChocolateMilk', 'black', 'Cyan', '2024-07-01', '2024-07-04', 3),
('ChocolateMilk', 'black', 'Azul', '2024-07-01', '2024-07-04', 1),
('ChocolateMilk', 'black', 'Navy', '2024-07-01', '2024-07-04', 2),
('ChocolateMilk', 'black', 'Tan', '2024-07-01', '2024-07-02', 3),
('ChocolateMilk', 'black', 'Susan', '2024-07-01', '2024-07-04', 1),
('ChocolateMilk', 'black', 'Bastet', '2024-07-01', '2024-07-04', 2),
('ChocolateMilk', 'black', 'Phoebe', '2024-07-01', '2024-07-04', 3),
('ChocolateMilk', 'black', 'Lilo', '2024-07-01', '2024-07-04', 4),
('ChocolateMilk', 'black', 'Indigo', '2024-07-01', '2024-07-02', 1),
('ChocolateMilk', 'black', 'Lilac', '2024-07-01', '2024-07-02', 3),
('ChocolateMilk', 'black', 'Violet', '2024-07-01', '2024-07-02', 4),
('ChocolateMilk', 'black', 'Sage', '2024-07-01', '2024-07-02', 1),
('ChocolateMilk', 'black', 'Sky', '2024-07-01', '2024-07-02', 3),
('ChocolateMilk', 'black', 'Pinky', '2024-07-01', '2024-07-02', 4),
('ChocolateMilk', 'black', 'Tan', '2024-07-03', '2024-07-04', 4),
('ChocolateMilk', 'black', 'Indigo', '2024-07-03', '2024-07-04', 1),
('ChocolateMilk', 'black', 'Violet', '2024-07-03', '2024-07-04', 2),
('ChocolateMilk', 'black', 'Lilac', '2024-07-03', '2024-07-04', 3),
('ChocolateMilk', 'black', 'Sage', '2024-07-03', '2024-07-04', 1),
('ChocolateMilk', 'black', 'Pinky', '2024-07-03', '2024-07-04', 2),
('ChocolateMilk', 'black', 'Sky', '2024-07-03', '2024-07-04', 4);

-- Insert data for ToyRAT task
INSERT INTO maze_map (task, genotype, animal_name, start_date, end_date, maze_number)
VALUES 
('ToyRAT', 'black', 'Joey', '2023-10-02', '2023-10-02', 2),
('ToyRAT', 'black', 'Bob', '2023-10-02', '2023-10-02', 3),
('ToyRAT', 'black', 'Teddy', '2023-10-02', '2023-10-02', 4),
('ToyRAT', 'black', 'Susan', '2023-10-03', '2023-10-19', 1),
('ToyRAT', 'black', 'Linda', '2023-10-03', '2023-10-19', 2),
('ToyRAT', 'black', 'Julie', '2023-10-03', '2023-10-19', 3),
('ToyRAT', 'black', 'Pheobe', '2023-10-03', '2023-10-19', 4),
('ToyRAT', 'black', 'Phoebe', '2023-10-01', '2023-10-30', 4),
('ToyRAT', 'black', 'Judy', '2023-10-03', '2023-10-26', 4),
('ToyRAT', 'black', 'Lilo', '2023-10-03', '2023-11-02', 4),
('ToyRAT', 'black', 'Teddy', '2023-10-16', '2023-10-18', 1),
('ToyRAT', 'black', 'Joey', '2023-10-16', '2023-10-18', 2),
('ToyRAT', 'black', 'Bob', '2023-10-16', '2023-10-18', 3),
('ToyRAT', 'black', 'Ross', '2023-10-16', '2023-10-18', 4),
('ToyRAT', 'black', 'Loki', '2023-10-16', '2023-11-01', 1),
('ToyRAT', 'black', 'Hermes', '2023-10-16', '2023-11-01', 2),
('ToyRAT', 'black', 'Gunther', '2023-10-18', '2023-11-01', 4),
('ToyRAT', 'black', 'Emma', '2023-10-31', '2023-11-27', 1),
('ToyRAT', 'black', 'Freya', '2023-10-31', '2023-11-27', 2),
('ToyRAT', 'black', 'Stitch', '2023-10-31', '2023-11-27', 3),
('ToyRAT', 'black', 'Bastet', '2023-10-31', '2023-11-27', 4),
('ToyRAT', 'black', 'Judy', '2023-11-02', '2023-11-02', 3);

-- Insert data for ToyStick task
INSERT INTO maze_map (task, genotype, animal_name, start_date, end_date, maze_number) 
VALUES 
('ToyStick', 'black', 'Susan', '2023-10-24', '2023-10-26', 1),
('ToyStick', 'black', 'Linda', '2023-10-03', '2023-10-26', 2),
('ToyStick', 'black', 'Julie', '2023-10-03', '2023-10-26', 3),
('ToyStick', 'black', 'Pheobe', '2023-10-03', '2023-11-02', 4),
('ToyStick', 'black', 'Judy', '2023-10-24', '2023-10-24', 4),
('ToyStick', 'black', 'Judy', '2023-10-31', '2023-11-09', 3),
('ToyStick', 'black', 'Lilo', '2023-10-24', '2023-11-09', 4),
('ToyStick', 'black', 'Teddy', '2023-10-25', '2023-11-15', 1),
('ToyStick', 'black', 'Joey', '2023-10-16', '2023-11-15', 2),
('ToyStick', 'black', 'Bob', '2023-10-16', '2023-11-15', 3),
('ToyStick', 'black', 'Ross', '2023-10-16', '2023-11-15', 4),
('ToyStick', 'black', 'Loki', '2023-11-06', '2023-11-15', 1),
('ToyStick', 'black', 'Hermes', '2023-11-06', '2023-11-15', 2),
('ToyStick', 'black', 'Gunther', '2023-11-06', '2023-11-15', 4);