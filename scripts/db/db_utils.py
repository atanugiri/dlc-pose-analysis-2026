from __future__ import annotations
from pathlib import Path
import pandas as pd

from scripts.config import DEFAULT_FPS, DB_CONNECT_KWARGS, DATA_DIR, EXCLUDED_IDS


def connect():
    """Create a DB connection using psycopg2."""
    try:
        import psycopg2
    except ImportError as exc:
        raise SystemExit(
            "psycopg2 is not installed in this Python environment. "
            "Install it with:\n"
            "conda install -n ghrelin -c conda-forge psycopg2\n"
            f"Import error: {exc}"
        )

    try:
        return psycopg2.connect(**DB_CONNECT_KWARGS)
    except Exception as exc:
        raise SystemExit(f"Database error: {exc}")

def fetch_ids_with_params(query: str, params: tuple) -> list[int]:
    """Run a parameterized query and return list of IDs."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def _apply_excluded_ids(record_ids: list[int]) -> list[int]:
    """Filter out IDs listed in EXCLUDED_IDS."""
    if not EXCLUDED_IDS:
        return record_ids
    return [record_id for record_id in record_ids if record_id not in EXCLUDED_IDS]
        
def get_filtered_pose_file(record_id: int) -> str:
    """Return experimental_metadata.filtered_pose_file for a given id."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filtered_pose_file
                FROM public.experimental_metadata
                WHERE id = %s
                """,
                (record_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or not row[0]:
        raise ValueError(f"No filtered_pose_file found for ID: {record_id}")

    return str(row[0])

def get_fps(record_id: int | None = None) -> float:
    """Return FPS for a record id, falling back to DEFAULT_FPS."""
    if record_id is None:
        return DEFAULT_FPS

    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT frame_rate
                    FROM public.experimental_metadata
                    WHERE id = %s
                    """,
                    (record_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if not row or row[0] is None:
            return DEFAULT_FPS

        fps = float(row[0])
        return fps if fps > 0 else DEFAULT_FPS

    except Exception:
        return DEFAULT_FPS

def get_frame_dimensions(record_id: int) -> tuple[int, int]:
    """Return (frame_width, frame_height) for a record ID."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT width, height
                FROM public.experimental_metadata
                WHERE id = %s
                """,
                (record_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or row[0] is None or row[1] is None:
        raise ValueError(f"No frame dimensions found for ID: {record_id}")

    return (int(row[0]), int(row[1]))

def get_maze_number(record_id: int) -> int | None:
    """Return maze_number for a record ID, or None when it is unavailable."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT maze_number
                FROM public.experimental_metadata
                WHERE id = %s
                """,
                (record_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or row[0] is None:
        return None

    return int(row[0])

def load_dlc_dataframe(filtered_pose_file: str) -> pd.DataFrame:
    """Load a filtered DLC file (H5 preferred, CSV fallback) from filtered_pose_data."""
    base_path = DATA_DIR / "filtered_pose_data" / filtered_pose_file
    h5_path = base_path.with_suffix(".h5")

    if h5_path.exists():
        return pd.read_hdf(h5_path, key="/df_with_missing")

    if base_path.exists():
        return pd.read_csv(base_path, header=[0, 1, 2], index_col=0)

    raise FileNotFoundError(f"File not found (tried .h5 and original): {base_path}")
