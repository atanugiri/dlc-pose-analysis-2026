from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from scripts.config import DB_CONNECT_KWARGS


def create_sqlalchemy_engine():
    host = DB_CONNECT_KWARGS["host"]
    port = DB_CONNECT_KWARGS["port"]
    user = DB_CONNECT_KWARGS["user"]
    database = DB_CONNECT_KWARGS["database"]

    db_url = f"postgresql+psycopg2://{user}@{host}:{port}/{database}"
    return create_engine(db_url)


def read_csv_with_dates(csv_path: Path, table_name: str) -> pd.DataFrame:
    parse_dates_map = {
        "experimental_metadata": ["session_date"],
        "maze_map": ["start_date", "end_date"],
    }
    return pd.read_csv(csv_path, parse_dates=parse_dates_map.get(table_name, None))


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
    args = parser.parse_args()

    csv_path = args.csv_file
    table_name = args.table

    engine = create_sqlalchemy_engine()

    print("Importing CSV file into PostgreSQL:")
    try:
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV: {csv_path}")

        print(f"\nImporting:")
        print(f"  CSV:   {csv_path}")
        print(f"  Table: {table_name}")

        df = read_csv_with_dates(csv_path, table_name)
        df.to_sql(
            table_name,
            engine,
            schema="public",
            if_exists="replace",
            index=False,
        )

        print(f"Done: {len(df)} rows x {len(df.columns)} columns")
    finally:
        engine.dispose()

    print("\nCSV import completed successfully.")


if __name__ == "__main__":
    main()
