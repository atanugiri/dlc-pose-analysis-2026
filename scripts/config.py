from pathlib import Path
import os
from dotenv import load_dotenv

# repo root = 2 levels above scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Select dotenv file via ENV_FILE (default: .env) and load from project root.
env_file_name = os.getenv("ENV_FILE", ".env")
load_dotenv(REPO_ROOT / env_file_name, override=False)

DATA_DIR = REPO_ROOT / "data"
paper_tag = os.getenv("PAPER_TAG", "").strip()
RESULTS_DIR = REPO_ROOT / "results"
if paper_tag:
    RESULTS_DIR = RESULTS_DIR / paper_tag
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

DEFAULT_FPS = float(os.getenv("DEFAULT_FPS", 15.0))
MAZE_SIZE_CM = float(os.getenv("MAZE_SIZE_CM", 64))

DB_CONNECT_KWARGS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER", "atanugiri"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "dlc_pose_analysis_2026"),
}
