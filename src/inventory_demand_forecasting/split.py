from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_rolling_features.csv"
)

TRAIN_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_train.csv"
)

TEST_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_test.csv"
)
