from __future__ import annotations

import argparse
import re
from collections.abc import Sequence

from scripts.db.db_utils import _apply_excluded_ids, connect, fetch_ids_with_params

SCHEMA_NAME = "public"
TABLE_NAME = "experimental_metadata"


def _parse_filter(value: str) -> tuple[str, object]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Filter must be in key=value format: {value!r}"
        )

    key, raw_value = value.split("=", 1)
    key = key.strip()
    raw_value = raw_value.strip()

    if not key:
        raise argparse.ArgumentTypeError(f"Filter key cannot be empty: {value!r}")

    if raw_value.lower() == "none" or raw_value == "":
        parsed_value: object = None
    elif re.fullmatch(r"[-+]?\d+", raw_value):
        parsed_value = int(raw_value)
    else:
        try:
            parsed_value = float(raw_value)
        except ValueError:
            parsed_value = raw_value

    return key, parsed_value


def _get_table_columns() -> set[str]:
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                """,
                (SCHEMA_NAME, TABLE_NAME),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def _build_query(filters: Sequence[tuple[str, object]]) -> tuple[str, tuple[object, ...]]:
    table_columns = _get_table_columns()
    where_clauses: list[str] = []
    params: list[object] = []

    for column_name, value in filters:
        if column_name not in table_columns:
            available = ", ".join(sorted(table_columns))
            raise ValueError(
                f"Unknown column {column_name!r}. Available columns: {available}"
            )
        if value is None:
            where_clauses.append(f"{column_name} IS NULL")
        else:
            where_clauses.append(f"{column_name} = %s")
            params.append(value)

    if not where_clauses:
        raise ValueError("At least one non-empty filter is required.")

    query = f"""
        SELECT id
        FROM {SCHEMA_NAME}.{TABLE_NAME}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY id;
    """
    return query, tuple(params)


def get_filtered_ids(filters: dict[str, object]) -> list[int]:
    """Return cleaned IDs for a dynamic filter mapping."""
    query, params = _build_query(list(filters.items()))
    ids = fetch_ids_with_params(query, params)
    return _apply_excluded_ids(ids)


def parse_filters(filter_args: Sequence[str]) -> dict[str, object]:
    """Parse repeated key=value tokens into a filter dict."""
    parsed: dict[str, object] = {}
    for raw_filter in filter_args:
        key, value = _parse_filter(raw_filter)
        parsed[key] = value
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Return cleaned ID list for experimental_metadata filters."
    )
    parser.add_argument(
        "--filter",
        action="append",
        type=_parse_filter,
        default=[],
        help="Repeat key=value filters; use None for SQL IS NULL matching.",
    )
    args = parser.parse_args()

    filters = dict(args.filter)
    ids = get_filtered_ids(filters)
    print(",".join(str(record_id) for record_id in ids))


if __name__ == "__main__":
    main()
