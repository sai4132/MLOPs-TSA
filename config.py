from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

AIR_PASSENGERS_CSV = RAW_DATA_DIR / "AirPassengers.csv"
AIR_PASSENGERS_DVC = RAW_DATA_DIR / "AirPassengers.csv.dvc"

REGISTRY_PATH = PROJECT_ROOT / "model_registry.yaml"
LOG_DIR = PROJECT_ROOT / "logs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
