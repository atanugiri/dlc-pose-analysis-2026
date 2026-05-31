from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

from scripts.config import DATA_DIR, DB_CONNECT_KWARGS


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
        description="Import project CSV files into PostgreSQL tables using pandas.to_sql."
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing CSV files (default: data/).",
    )
    args = parser.parse_args()

    csv_to_table = {
        "experimental_metadata.csv": "experimental_metadata",
        "maze_map.csv": "maze_map",
    }

    engine = create_sqlalchemy_engine()

    print("Importing CSV files into PostgreSQL:")
    try:
        for csv_name, table_name in csv_to_table.items():
            csv_path = args.csv_dir / csv_name
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
