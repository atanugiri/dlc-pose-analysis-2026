from __future__ import annotations

import numpy as np
import pandas as pd


def get_bodypart_xy(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    individual: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract x, y, and likelihood for one bodypart from a DLC-style DataFrame."""
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected DLC-style MultiIndex columns.")

    scorer = df.columns.get_level_values(0).unique().tolist()[0]

    if df.columns.nlevels >= 4:
        if individual is None:
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(
                "Multi-animal data requires `individual`. "
                f"Available individuals: {available}"
            )

        if individual not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Individual {individual!r} not found. Available: {available}")

        coords = df[scorer][individual][bodypart]

    else:
        if bodypart not in df[scorer].columns.get_level_values(0):
            available = df[scorer].columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Bodypart {bodypart!r} not found. Available: {available}")

        coords = df[scorer][bodypart]

    if not {"x", "y", "likelihood"}.issubset(coords.columns):
        raise ValueError(
            f"Bodypart {bodypart!r} is missing x/y/likelihood columns. Found: {list(coords.columns)}"
        )

    x = coords["x"].astype(float).to_numpy()
    y = coords["y"].astype(float).to_numpy()
    likelihood = coords["likelihood"].astype(float).to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(likelihood)

    return x[valid], y[valid], likelihood[valid]


def get_bodypart_xy_time(
    df: pd.DataFrame,
    *,
    bodypart: str = "Midback",
    fps: float,
    individual: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract x, y, likelihood, and time arrays for one bodypart."""
    x, y, likelihood = get_bodypart_xy(
        df,
        bodypart=bodypart,
        individual=individual,
    )

    time = np.arange(len(x)) / fps

    return x, y, likelihood, time