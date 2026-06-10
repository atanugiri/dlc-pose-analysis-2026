from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import re
import sys

import pandas as pd
from sqlalchemy import create_engine, text

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

DB_CONNECT_KWARGS = importlib.import_module("scripts.config").DB_CONNECT_KWARGS


def create_sqlalchemy_engine():
    host = DB_CONNECT_KWARGS["host"]
    port = DB_CONNECT_KWARGS["port"]
    user = DB_CONNECT_KWARGS["user"]
    database = DB_CONNECT_KWARGS["database"]

    db_url = f"postgresql+psycopg2://{user}@{host}:{port}/{database}"
    print(f"Connecting to database with URL: {db_url}")
    return create_engine(db_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import one CSV file into PostgreSQL using pandas.to_sql."
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        required=True,
        help="Full path to CSV file to import.",
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Target table name (for example: experimental_metadata or maze_map).",
    )
    parser.add_argument(
        "--mode",
        choices=["replace", "append", "truncate_append"],
        default="truncate_append",
        help=(
            "Import mode: replace drops/recreates table; append inserts rows; "
            "truncate_append clears table then appends (preserves schema)."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    table_name = args.table
    mode = args.mode

    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(f"Unsafe table name: {table_name!r}")

    engine = create_sqlalchemy_engine()

    print("Importing CSV file into PostgreSQL:")
    try:
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV: {csv_path}")

        print(f"\nImporting:")
        print(f"  CSV:   {csv_path}")
        print(f"  Table: {table_name}")

        df = pd.read_csv(csv_path)
        if mode == "truncate_append":
            with engine.begin() as conn:
                conn.execute(text(f'TRUNCATE TABLE public."{table_name}" RESTART IDENTITY'))
            if_exists = "append"
        elif mode == "append":
            if_exists = "append"
        else:
            if_exists = "replace"

        df.to_sql(table_name, engine, schema="public", if_exists=if_exists, index=False)

        print(f"Done: {len(df)} rows x {len(df.columns)} columns (mode={mode})")
    finally:
        engine.dispose()

    print("\nCSV import completed successfully.")


if __name__ == "__main__":
    main()
