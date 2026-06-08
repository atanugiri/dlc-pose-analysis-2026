from __future__ import annotations

import argparse

import scripts.db.db_utils as db_utils
from scripts.features.estimate_maze_corners import estimate_maze_corners_from_ids


def fetch_all_ids() -> list[int]:
    conn = db_utils.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM public.experimental_metadata ORDER BY id")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def update_corners(record_id: int, corners: dict) -> None:
    conn = db_utils.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE public.experimental_metadata
                SET maze_corners = %s
                WHERE id = %s
                """,
                (
                    [
                        float(corners["x_min"]),
                        float(corners["x_max"]),
                        float(corners["y_min"]),
                        float(corners["y_max"]),
                    ],
                    record_id,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate and inject maze corners into experimental_metadata per record."
    )
    parser.add_argument("--quantile-low", type=float, default=0.1)
    parser.add_argument("--quantile-high", type=float, default=99.9)
    parser.add_argument("--likelihood-threshold", type=float, default=0.9)
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument(
        "--individual",
        default=None,
        help="DLC individual name for multi-animal files (e.g., m1).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when any record fails.",
    )
    args = parser.parse_args()

    ids = fetch_all_ids()
    quantiles = (float(args.quantile_low), float(args.quantile_high))

    updated = 0
    skipped = 0

    for record_id in ids:
        try:
            corners = estimate_maze_corners_from_ids(
                record_id,
                quantiles=quantiles,
                individual=args.individual,
                likelihood_threshold=args.likelihood_threshold,
                smoothing_window=args.smoothing_window,
            )
            update_corners(record_id, corners)
            updated += 1
        except Exception as exc:
            skipped += 1
            print(f"Skipping id={record_id}: {exc}")
            if args.stop_on_error:
                raise

    print(f"Done. Updated={updated}, Skipped={skipped}")


if __name__ == "__main__":
    main()
