
"""Business logic for generating demand forecasts with the trained XGBoost model."""

from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

# Find the project root from this module's location.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define the paths to the saved model and cleaned historical sales.
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"
SALES_PATH = PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"

# Keep the feature order consistent with the training pipeline.
FEATURE_COLUMNS = [
    "store",
    "item",
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


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model once and reuse it across API requests."""

    # Fail clearly if the model has not been trained or saved.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found: {MODEL_PATH}"
        )

    # The artifact is a fitted sklearn pipeline containing preprocessing
    # and the trained XGBoost estimator.
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_sales_history() -> pd.DataFrame:
    """Load and cache the cleaned historical sales dataset."""

    # Check that the historical dataset is available.
    if not SALES_PATH.is_file():
        raise FileNotFoundError(
            f"Historical sales dataset not found: {SALES_PATH}"
        )

    # Load the sales records and parse dates as datetime values.
    sales = pd.read_csv(
        SALES_PATH,
        parse_dates=["date"],
    )

    # Confirm that the columns needed for forecasting exist.
    required_columns = {"date", "store", "item", "sales"}
    missing_columns = required_columns.difference(sales.columns)

    if missing_columns:
        raise ValueError(
            f"Historical sales data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Sort records so that lag and rolling calculations use chronological
    # history for each store-item combination.
    sales = sales.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    return sales


def build_forecast_features(
    store: int,
    item: int,
    forecast_date: date,
) -> pd.DataFrame:
    """Build the 14 model features using only sales before forecast_date."""

    # Reject identifiers outside the dataset's known range.
    if not 1 <= store <= 10:
        raise ValueError("Store must be between 1 and 10.")

    if not 1 <= item <= 50:
        raise ValueError("Item must be between 1 and 50.")

    # Convert the requested date to a pandas Timestamp for calendar features.
    target_date = pd.Timestamp(forecast_date)

    # Load the cleaned sales history.
    sales = load_sales_history()

    # Select only the requested store-item series.
    history = sales.loc[
        (sales["store"] == store)
        & (sales["item"] == item)
        & (sales["date"] < target_date)
    ].sort_values("date")

    # Thirty previous observations are required for the 30-day rolling mean.
    if len(history) < 30:
        raise ValueError(
            "At least 30 days of historical sales are required "
            "before the forecast date."
        )

    # Use the latest 30 observations that occurred before the forecast date.
    recent_30 = history.tail(30)

    # Confirm these observations are exactly the 30 calendar days before
    # the requested date. This prevents silently using stale or incomplete
    # history when the dataset has gaps.
    expected_dates = pd.date_range(
        end=target_date - pd.Timedelta(days=1),
        periods=30,
        freq="D",
    )

    if not recent_30["date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="date")
    ):
        raise ValueError(
            "The 30 days immediately before the forecast date must have "
            "complete historical sales. Check the requested date and data."
        )

    # Extract the historical sales values in chronological order.
    previous_sales = recent_30["sales"]

    # ISO week numbering matches the week-of-year feature used during training.
    iso_week = target_date.isocalendar().week

    # Construct one row with exactly the features expected by the model.
    feature_row = {
        "store": store,
        "item": item,
        "day_of_week": target_date.dayofweek,
        "day_of_month": target_date.day,
        "month": target_date.month,
        "year": target_date.year,
        "quarter": target_date.quarter,
        "week_of_year": int(iso_week),
        "is_weekend": int(target_date.dayofweek >= 5),

        # Lag 1 is the most recent actual sales observation.
        "sales_lag_1": float(previous_sales.iloc[-1]),

        # Lag 7 is the actual sales value seven days before the target date.
        "sales_lag_7": float(previous_sales.iloc[-7]),

        # Calculate the rolling mean from the previous seven actual values.
        "sales_rolling_mean_7": float(previous_sales.tail(7).mean()),

        # Calculate the rolling mean from the previous thirty actual values.
        "sales_rolling_mean_30": float(previous_sales.mean()),

        # pandas uses sample standard deviation by default (ddof=1).
        "sales_rolling_std_7": float(previous_sales.tail(7).std()),
    }

    # Preserve the exact column order expected by the trained pipeline.
    return pd.DataFrame(
        [feature_row],
        columns=FEATURE_COLUMNS,
    )


def forecast_sales(
    store: int,
    item: int,
    forecast_date: date,
) -> dict:
    """Generate a forecast and return the prediction with its metadata."""

    # The saved model is loaded once and reused for subsequent predictions.
    model = load_model()

    # Build features from historical sales and the requested calendar date.
    features = build_forecast_features(
        store=store,
        item=item,
        forecast_date=forecast_date,
    )

    # Predict demand for this single store-item-date combination.
    prediction = float(model.predict(features)[0])

    # Demand cannot be negative, so clamp negative model outputs to zero.
    prediction = max(0.0, prediction)

    # Return a structured result that can be used by the API response schema.
    return {
        "store": store,
        "item": item,
        "date": forecast_date,
        "predicted_sales": prediction,
        "model_name": "XGBoost",
    }
