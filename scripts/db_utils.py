from __future__ import annotations

DEFAULT_FPS = 15.0

DB_CONNECT_KWARGS = {
    "host": "localhost",
    "port": 5432,
    "user": "atanugiri",
    "password": "",
    "database": "dlc_pose_analysis_2026",
}


def _connect():
    """Create a DB connection (psycopg2)."""
    try:
        import psycopg2
    except ImportError as exc:
        raise SystemExit(
            "psycopg2 is not installed in this Python environment. "
            "Install it (e.g. `conda install -n ghrelin -c conda-forge psycopg2`).\n"
            f"Import error: {exc}"
        )

    try:
        return psycopg2.connect(**DB_CONNECT_KWARGS)
    except Exception as exc:
        raise SystemExit(f"Database error: {exc}")

def fetch_ids(query):
    """Run a query and return list of IDs."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()

def get_filtered_pose_file(record_id: int) -> str:
    """Return experimental_metadata.filtered_pose_file for a given id."""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT filtered_pose_file FROM public.experimental_metadata WHERE id = %s",
        (record_id,),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row or not row[0]:
        raise ValueError(f"No filtered_pose_file found for ID: {record_id}")
    return str(row[0])


def get_fps(record_id: int | None = None) -> float:
    """Return FPS for a record id from the database, falling back to DEFAULT_FPS.

    Notes
    -----
    - If the `frame_rate` column doesn't exist yet (or is NULL), returns DEFAULT_FPS.
    - If DB connectivity isn't available, returns DEFAULT_FPS.
    """
    if record_id is None:
        return DEFAULT_FPS

    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT frame_rate FROM public.experimental_metadata WHERE id = %s",
            (record_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row or row[0] is None:
            return DEFAULT_FPS

        fps = float(row[0])
        if fps <= 0:
            return DEFAULT_FPS
        return fps
    except Exception:
        return DEFAULT_FPS

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get filtered_pose_file for a given record ID.")
    parser.add_argument("record_id", type=int, help="Record ID to query")
    args = parser.parse_args()

    try:
        filtered_pose_file = get_filtered_pose_file(args.record_id)
        print(f"Filtered pose file for ID {args.record_id}: {filtered_pose_file}")
        fps = get_fps(args.record_id)
        print(f"FPS for ID {args.record_id}: {fps}")
    except Exception as exc:
        print(f"Error: {exc}")