from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"