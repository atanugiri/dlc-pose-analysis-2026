from pathlib import Path

# repo root = 2 levels above scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

DEFAULT_FPS = 15.0

DB_CONNECT_KWARGS = {
    "host": "localhost",
    "port": 5432,
    "user": "atanugiri",
    "password": "",
    "database": "dlc_pose_analysis_2026",
}
