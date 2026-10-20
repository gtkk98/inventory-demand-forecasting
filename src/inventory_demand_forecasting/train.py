
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from inventory_demand_forecasting.evaluate import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "sales_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "sales_test.csv"

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "linear_regression_predictions.csv"
)
METRICS_PATH = PROJECT_ROOT / "reports" / "linear_regression_metrics.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression.joblib"


CATEGORICAL_FEATURES = ["store", "item"]

NUMERIC_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "quarter",
    "week_of_year",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_mean_7",
    "sales_rolling_mean_30",
    "sales_rolling_std_7",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "sales"


def build_model() -> Pipeline:
    """Build a preprocessing and Linear Regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def prepare_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select valid rows and separate model features from the target."""

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy()

    numeric_values = result[NUMERIC_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Reject invalid or infinite feature values instead of silently
    # allowing corrupted data into model training.
    if not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
        raise ValueError(
            "Numeric features contain missing, invalid, or infinite values. "
            "Filter expected historical warm-up rows before calling this "
            "function."
        )

    if result[CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("Store and item features must not be missing.")

    target = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")

    if not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("The target contains missing or invalid values.")

    if (target < 0).any():
        raise ValueError("Sales targets must not be negative.")

    return result[FEATURE_COLUMNS], target.astype(float)


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}. "
            "Run the chronological split first."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}. "
            "Run the chronological split first."
        )

    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    test_df = pd.read_csv(TEST_PATH, parse_dates=["date"])

    # The first 30 days of each series lack enough history for the
    # 30-day rolling mean. Drop those rows from training only.
    train_df = train_df.dropna(subset=NUMERIC_FEATURES).copy()

    X_train, y_train = prepare_training_data(train_df)
    X_test, y_test = prepare_training_data(test_df)

    # Safety check: the test period must occur after the training period.
    if train_df["date"].max() >= test_df["date"].min():
        raise ValueError("Training and testing dates overlap.")

    model = build_model()
    model.fit(X_train, y_train)

    # Sales cannot be negative, so clamp negative estimates to zero.
    predictions = np.maximum(model.predict(X_test), 0)

    metrics = calculate_metrics(
        y_test,
        pd.Series(predictions, index=y_test.index),
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                **metrics,
            }
        ]
    )

    prediction_df = test_df[
        ["date", "store", "item", "sales"]
    ].copy()

    prediction_df["linear_regression_prediction"] = predictions

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    prediction_df.to_csv(PREDICTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    print("\n=== LINEAR REGRESSION TRAINING ===")
    print(f"Training rows used: {len(X_train):,}")
    print(f"Testing rows used:  {len(X_test):,}")
    print(f"Training end date:  {train_df['date'].max().date()}")
    print(f"Testing start date: {test_df['date'].min().date()}")

    print("\n=== MODEL METRICS ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== PREDICTION PREVIEW ===")
    print(prediction_df.head(10).to_string(index=False))

    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to:     {METRICS_PATH}")
    print(f"Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
