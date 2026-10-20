
from pathlib import Path

import pandas as pd

from inventory_demand_forecasting.evaluate import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_test.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_predictions.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "baseline_metrics.csv"
)


def create_baseline_predictions(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create previous-day and seasonal-naive predictions."""

    required_columns = {
        "date",
        "store",
        "item",
        "sales",
        "sales_lag_1",
        "sales_lag_7",
    }

    missing_columns = required_columns.difference(test_df.columns)

    if missing_columns:
        raise ValueError(
            f"Test dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if test_df.empty:
        raise ValueError("The test dataset is empty.")

    result = test_df.copy()

    if result[
        ["sales", "sales_lag_1", "sales_lag_7"]
    ].isna().any().any():
        raise ValueError(
            "Missing actual or lag values found. "
            "Resolve them before calculating baseline predictions."
        )

    result["naive_lag_1_prediction"] = result["sales_lag_1"]
    result["seasonal_naive_lag_7_prediction"] = result["sales_lag_7"]

    return result[
        [
            "date",
            "store",
            "item",
            "sales",
            "naive_lag_1_prediction",
            "seasonal_naive_lag_7_prediction",
        ]
    ]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {INPUT_PATH}. "
            "Run the chronological split first."
        )

    test_df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    predictions = create_baseline_predictions(test_df)

    metrics_rows = []

    model_columns = {
        "Previous-day naive": "naive_lag_1_prediction",
        "Seasonal naive (7 days)": "seasonal_naive_lag_7_prediction",
    }

    for model_name, prediction_column in model_columns.items():
        metrics = calculate_metrics(
            predictions["sales"],
            predictions[prediction_column],
        )

        metrics_rows.append(
            {
                "model": model_name,
                "test_rows": len(predictions),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\n=== BASELINE FORECASTING RESULTS ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== PREDICTION PREVIEW ===")
    print(predictions.head(10).to_string(index=False))

    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to:     {METRICS_PATH}")


if __name__ == "__main__":
    main()
