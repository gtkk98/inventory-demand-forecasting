
"""Validate XGBoost artifacts, predictions, and saved-model reproducibility."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from inventory_demand_forecasting.xgboost_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "sales_test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "xgboost_predictions.csv"
)
METRICS_PATH = PROJECT_ROOT / "reports" / "xgboost_metrics.csv"


def main() -> None:
    """Check saved model, predictions, and evaluation report."""

    required_paths = [
        TEST_PATH,
        MODEL_PATH,
        PREDICTIONS_PATH,
        METRICS_PATH,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    test_df = pd.read_csv(TEST_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)

    print("=== XGBOOST ARTIFACT VALIDATION ===")

    # 1. Verify that every test row has a prediction.
    assert len(predictions_df) == len(test_df), (
        "Prediction row count does not match test row count."
    )
    print(f"PASS: prediction row count = {len(predictions_df):,}")

    # 2. Verify required prediction columns.
    required_prediction_columns = {
        "date",
        "store",
        "item",
        "actual_sales",
        "predicted_sales",
    }

    missing_columns = (
        required_prediction_columns - set(predictions_df.columns)
    )

    assert not missing_columns, (
        f"Prediction report is missing columns: {sorted(missing_columns)}"
    )
    print("PASS: prediction report contains all required columns")

    # 3. Verify that predictions are finite and non-negative.
    actual = predictions_df["actual_sales"].to_numpy(dtype=float)
    predicted = predictions_df["predicted_sales"].to_numpy(dtype=float)

    assert np.isfinite(actual).all(), "Actual sales contain non-finite values."
    assert np.isfinite(predicted).all(), (
        "Predictions contain non-finite values."
    )
    assert (predicted >= 0).all(), "Predictions contain negative sales."
    print("PASS: actual sales and predictions are finite")
    print("PASS: predictions are non-negative")

    # 4. Verify that the date/store/item keys are unique.
    keys = ["date", "store", "item"]
    assert not predictions_df.duplicated(keys).any(), (
        "Prediction report contains duplicate date/store/item rows."
    )
    print("PASS: prediction keys are unique")

    # 5. Verify the metrics report.
    required_metric_columns = {
        "model",
        "train_rows",
        "test_rows",
        "MAE",
        "RMSE",
        "WAPE_percent",
    }

    missing_metrics = required_metric_columns - set(metrics_df.columns)

    assert not missing_metrics, (
        f"Metrics report is missing columns: {sorted(missing_metrics)}"
    )
    assert len(metrics_df) == 1, (
        "Expected exactly one row in the XGBoost metrics report."
    )

    metrics = metrics_df.iloc[0]

    assert metrics["model"] == "XGBoost", "Unexpected model name."
    assert int(metrics["test_rows"]) == len(test_df), (
        "Metrics test row count does not match test data."
    )

    calculated_mae = np.mean(np.abs(actual - predicted))
    calculated_rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    calculated_wape = (
        np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    )

    np.testing.assert_allclose(
        [metrics["MAE"], metrics["RMSE"], metrics["WAPE_percent"]],
        [calculated_mae, calculated_rmse, calculated_wape],
        rtol=1e-4,
        atol=1e-4,
    )
    print("PASS: saved metrics match the prediction report")

    # 6. Reload the serialized model and reproduce sample predictions.
    pipeline = joblib.load(MODEL_PATH)

    sample_size = min(100, len(test_df))
    sample_features = test_df.iloc[:sample_size][FEATURE_COLUMNS]

    reloaded_predictions = pipeline.predict(sample_features)
    saved_predictions = predicted[:sample_size]

    np.testing.assert_allclose(
        reloaded_predictions,
        saved_predictions,
        rtol=1e-5,
        atol=1e-5,
    )
    print(
        f"PASS: reloaded model reproduces the first "
        f"{sample_size} saved predictions"
    )

    print("\nAll XGBoost artifact checks passed.")


if __name__ == "__main__":
    main()
