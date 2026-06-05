from pathlib import Path
import os
from dotenv import load_dotenv

# repo root = 2 levels above scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Select dotenv file via ENV_FILE (default: .env) and load from project root.
env_file_name = os.getenv("ENV_FILE", ".env")
load_dotenv(REPO_ROOT / env_file_name, override=False)


def _parse_int_set(csv_value: str) -> set[int]:
    values: set[int] = set()
    for token in csv_value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in EXCLUDED_IDS: {token!r}") from exc
    return values

paper_tag = os.getenv("PAPER_TAG", "").strip()
base_data_dir = REPO_ROOT / "data"
paper_data_dir = base_data_dir / paper_tag if paper_tag else base_data_dir
DATA_DIR = paper_data_dir if paper_data_dir.exists() else base_data_dir

base_results_dir = REPO_ROOT / "results"
results_dir = base_results_dir / paper_tag if paper_tag else base_results_dir
RESULTS_DIR = results_dir if results_dir.exists() else base_results_dir
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

DEFAULT_FPS = float(os.getenv("DEFAULT_FPS", 15.0))
MAZE_SIZE_CM = float(os.getenv("MAZE_SIZE_CM", 64))
EXCLUDED_IDS = _parse_int_set(os.getenv("EXCLUDED_IDS", ""))

DB_CONNECT_KWARGS = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER", "atanugiri"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "dlc_pose_analysis_2026"),
}
