from pathlib import Path

# repo root = 2 levels above scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"