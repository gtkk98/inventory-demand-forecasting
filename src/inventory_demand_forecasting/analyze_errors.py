
"""Analyze XGBoost forecast errors and generate diagnostic reports."""
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "xgboost_predictions.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

ERROR_DETAILS_PATH = REPORTS_DIR / "xgboost_error_analysis.csv"
SUMMARY_PATH = REPORTS_DIR / "xgboost_error_summary.csv"


REQUIRED_COLUMNS = [
    "date",
    "store",
    "item",
    "actual_sales",
    "predicted_sales",
]


def load_predictions(path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    """Load and validate saved XGBoost predictions."""

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. "
            "Run train_xgboost first."
        )

    df = pd.read_csv(path)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Prediction file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Prediction file contains no rows.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Prediction file contains invalid dates.")

    for column in ["actual_sales", "predicted_sales"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Column '{column}' contains non-finite values."
            )

    if (df["actual_sales"] < 0).any():
        raise ValueError("Actual sales cannot be negative.")

    if (df["predicted_sales"] < 0).any():
        raise ValueError("Predicted sales cannot be negative.")

    if df.duplicated(["date", "store", "item"]).any():
        raise ValueError("Duplicate date/store/item prediction rows.")

    return df


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate row-level forecasting errors."""

    result = df.copy()

    result["error"] = (
        result["actual_sales"] - result["predicted_sales"]
    )

    result["absolute_error"] = result["error"].abs()

    result["squared_error"] = result["error"] ** 2

    result["absolute_percentage_error"] = np.where(
        result["actual_sales"] != 0,
        result["absolute_error"] / result["actual_sales"] * 100,
        np.nan,
    )

    result["month"] = result["date"].dt.month

    result["day_of_week"] = result["date"].dt.dayofweek

    result["day_name"] = result["date"].dt.day_name()

    return result


def build_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate overall forecast error metrics."""

    actual = df["actual_sales"]
    predicted = df["predicted_sales"]
    errors = df["error"]

    total_actual = actual.sum()

    wape = (
        df["absolute_error"].sum() / total_actual * 100
        if total_actual != 0
        else np.nan
    )

    summary = {
        "prediction_rows": len(df),
        "MAE": df["absolute_error"].mean(),
        "RMSE": np.sqrt(df["squared_error"].mean()),
        "WAPE_percent": wape,
        "mean_error": errors.mean(),
        "median_error": errors.median(),
        "underprediction_rows": int((errors > 0).sum()),
        "overprediction_rows": int((errors < 0).sum()),
        "exact_prediction_rows": int((errors == 0).sum()),
        "actual_sales_total": total_actual,
        "predicted_sales_total": predicted.sum(),
    }

    return pd.DataFrame([summary])


def main() -> None:
    """Run error analysis and save reports."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    error_details = add_error_columns(predictions)
    summary = build_error_summary(error_details)

    error_details.to_csv(ERROR_DETAILS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n=== XGBOOST FORECAST ERROR SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nDetailed errors saved to: {ERROR_DETAILS_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
