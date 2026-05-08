from __future__ import annotations

import numpy as np

from scripts.utils.dlc_utils import get_bodypart_xy_time
import scripts.db.db_utils as db_utils


def estimate_maze_corners_from_id(
    record_id: int,
    *,
    quantiles: tuple[float, float] = (0.5, 99.5),
    individual: str | None = None,
    likelihood_threshold: float | None = None,
    smoothing_window: int | None = None,
) -> dict:
    """Estimate maze extents (robust corners) from Midback positions for one record.

    Loads the filtered pose file for ``record_id``, extracts valid Midback x/y
    samples via ``get_bodypart_xy_time``, and returns the requested lower/upper
    quantiles per axis as the maze corners.
    """
    filtered_pose_file = db_utils.get_filtered_pose_file(record_id)
    df = db_utils.load_dlc_dataframe(filtered_pose_file)
    fps = db_utils.get_fps(record_id)

    qlow, qhigh = float(quantiles[0]), float(quantiles[1])
    x, y, _, _, _ = get_bodypart_xy_time(
        df,
        bodypart="Midback",
        fps=float(fps),
        individual=individual,
        smoothing_window=smoothing_window,
        likelihood_threshold=likelihood_threshold,
    )

    if x.size == 0 or y.size == 0:
        raise ValueError(f"No valid Midback samples for record {record_id}.")

    x_min = float(np.nanpercentile(x, qlow))
    x_max = float(np.nanpercentile(x, qhigh))
    y_min = float(np.nanpercentile(y, qlow))
    y_max = float(np.nanpercentile(y, qhigh))
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
    return dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, corners=corners)


def estimate_maze_corners_from_ids(
    record_id: int,
    *,
    quantiles: tuple[float, float] = (0.5, 99.5),
    individual: str | None = None,
    likelihood_threshold: float | None = None,
    smoothing_window: int | None = None,
) -> dict:
    """Estimate maze corners pooled across all trials sharing the same maze_number and task.

    Queries the DB for the ``maze_number`` and ``task`` of ``record_id``, then
    concatenates Midback x/y from every record with matching values before
    computing quantile-based corners.  Raises ``ValueError`` if no
    ``maze_number`` is found for the record.
    """
    conn = db_utils.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT maze_number, task FROM public.experimental_metadata WHERE id = %s",
                (record_id,),
            )
            row = cur.fetchone()
            if not row or row[0] is None or row[1] is None:
                raise ValueError(
                    f"record {record_id} has no maze_number/task; cannot pool across maze."
                )
            maze_number, task = row[0], row[1]
            cur.execute(
                "SELECT id FROM public.experimental_metadata"
                " WHERE maze_number = %s AND task = %s",
                (maze_number, task),
            )
            ids = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()

    samples_x: list[np.ndarray] = []
    samples_y: list[np.ndarray] = []
    for rid in ids:
        try:
            fp = db_utils.get_filtered_pose_file(rid)
            df = db_utils.load_dlc_dataframe(fp)
            fps = db_utils.get_fps(rid)
            xr, yr, _, _, _ = get_bodypart_xy_time(
                df,
                bodypart="Midback",
                fps=float(fps),
                individual=individual,
                smoothing_window=smoothing_window,
                likelihood_threshold=likelihood_threshold,
            )
            if xr.size:
                samples_x.append(xr)
                samples_y.append(yr)
        except Exception:
            continue

    if not samples_x:
        raise ValueError(
            f"No valid Midback samples found for maze_number={maze_number}, task={task!r}."
        )

    allx = np.concatenate(samples_x)
    ally = np.concatenate(samples_y)
    qlow, qhigh = float(quantiles[0]), float(quantiles[1])
    x_min = float(np.nanpercentile(allx, qlow))
    x_max = float(np.nanpercentile(allx, qhigh))
    y_min = float(np.nanpercentile(ally, qlow))
    y_max = float(np.nanpercentile(ally, qhigh))
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
    return dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, corners=corners)
