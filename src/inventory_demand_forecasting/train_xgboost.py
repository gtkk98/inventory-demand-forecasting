
"""Train and evaluate the XGBoost demand forecasting model."""

from pathlib import Path

import joblib
import pandas as pd

from inventory_demand_forecasting.evaluate import calculate_metrics
from inventory_demand_forecasting.xgboost_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_xgboost_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "sales_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "sales_test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "xgboost_predictions.csv"
)
METRICS_PATH = PROJECT_ROOT / "reports" / "xgboost_metrics.csv"


def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Check that a dataset contains the columns required for training."""

    required_columns = [
        "date",
        "store",
        "item",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

    if df["date"].isna().any():
        raise ValueError(f"{dataset_name} contains missing dates.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing target values."
        )

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            f"{dataset_name} contains missing feature values."
        )

    numeric_columns = [
        *[
            column
            for column in FEATURE_COLUMNS
            if column not in ["store", "item"]
        ],
        TARGET_COLUMN,
    ]

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="raise")

        if not values.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        ).all():
            raise ValueError(
                f"{dataset_name} contains non-finite values "
                f"in column '{column}'."
            )


def main() -> None:
    """Load data, train XGBoost, predict, and save results."""

    for path in [TRAIN_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}. "
                "Run the data preparation and split scripts first."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("=== XGBOOST TRAINING ===")
    print(f"Training rows loaded: {len(train_df):,}")
    print(f"Test rows loaded: {len(test_df):,}")

    # Rolling features may be missing at the beginning of each series.
    train_df = train_df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    validate_dataset(train_df, "Training dataset")
    validate_dataset(test_df, "Test dataset")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print(f"Training rows used: {len(X_train):,}")
    print(f"Test rows used: {len(X_test):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print("\nTraining model. This may take several minutes...")

    pipeline = build_xgboost_pipeline()
    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Predictions are generated in the original sales units.
    predictions = pipeline.predict(X_test)

    # Demand cannot be negative.
    predictions = predictions.clip(min=0)

    metrics = calculate_metrics(y_test, predictions)

    prediction_report = test_df[
        ["date", "store", "item", TARGET_COLUMN]
    ].copy()

    prediction_report = prediction_report.rename(
        columns={TARGET_COLUMN: "actual_sales"}
    )
    prediction_report["predicted_sales"] = predictions

    metrics_report = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                **metrics,
            }
        ]
    )

    # Create output directories if they do not exist.
    for path in [MODEL_PATH, PREDICTIONS_PATH, METRICS_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    prediction_report.to_csv(PREDICTIONS_PATH, index=False)
    metrics_report.to_csv(METRICS_PATH, index=False)

    print("\n=== XGBOOST RESULTS ===")
    print(
        metrics_report.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
