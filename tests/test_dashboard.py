
"""Tests for the dashboard's data-processing and validation helpers."""

# Import NumPy to create and verify numeric test values.
import numpy as np

# Import pandas to build small test DataFrames.
import pandas as pd

# Import pytest to check that invalid data raises expected errors.
import pytest

# Import the dashboard helper functions we want to test.
from inventory_demand_forecasting.dashboard import (
    validate_model_comparison,
    prepare_prediction_chart,
    prepare_daily_error_chart,
    filter_predictions,
)


def test_validate_model_comparison_ranks_best_wape_first():
    """The model with the lowest WAPE should rank first."""

    # Create a small example containing two forecasting models.
    models = pd.DataFrame(
        {
            "model": ["Linear Regression", "XGBoost"],
            "MAE": [7.0, 6.0],
            "RMSE": [9.0, 8.0],
            "WAPE_percent": [12.0, 10.0],
        }
    )

    # Run the same validation and ranking function used by the dashboard.
    result = validate_model_comparison(models)

    # Confirm that XGBoost ranks first because its WAPE is lower.
    assert result.iloc[0]["model"] == "XGBoost"

    # Confirm that ranking preserves the expected metric value.
    assert result.iloc[0]["WAPE_percent"] == 10.0


def test_validate_model_comparison_rejects_missing_columns():
    """A report missing required columns should be rejected."""

    # Deliberately omit the RMSE and WAPE columns.
    invalid_models = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "MAE": [6.0],
        }
    )

    # Validation should fail rather than silently accepting incomplete data.
    with pytest.raises(ValueError):
        validate_model_comparison(invalid_models)


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.inf,       # Positive infinity is not a valid metric.
        -np.inf,      # Negative infinity is not a valid metric.
        np.nan,       # A missing numeric metric is not valid either.
        -1.0,         # Error metrics cannot be negative.
    ],
)
def test_validate_model_comparison_rejects_invalid_metrics(invalid_value):
    """Invalid error metrics should not appear in the model report."""

    # Build a report with one deliberately invalid WAPE value.
    models = pd.DataFrame(
        {
            "model": ["XGBoost"],
            "MAE": [6.0],
            "RMSE": [8.0],
            "WAPE_percent": [invalid_value],
        }
    )

    # The dashboard should raise a ValueError for invalid metrics.
    with pytest.raises(ValueError):
        validate_model_comparison(models)


def test_prepare_prediction_chart_aggregates_daily_totals():
    """Sales from different store-item pairs should be summed by date."""

    # Create two store-item predictions for each of two dates.
    predictions = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
            ],
            "store": [1, 2, 1, 2],
            "item": [1, 1, 1, 1],
            "actual_sales": [10, 20, 15, 25],
            "predicted_sales": [11, 19, 14, 26],
        }
    )

    # Aggregate individual series into one actual and predicted total per day.
    result = prepare_prediction_chart(predictions)

    # Confirm that both dates remain in the aggregated output.
    assert len(result) == 2

    # Confirm the actual daily totals: 10 + 20, then 15 + 25.
    assert result["actual_sales"].tolist() == [30, 40]

    # Confirm the predicted daily totals: 11 + 19, then 14 + 26.
    assert result["predicted_sales"].tolist() == [30, 40]


def test_prepare_daily_error_chart_calculates_errors():
    """Signed error and absolute error should be calculated correctly."""

    # Create two observations with known actual and predicted values.
    predictions = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "actual_sales": [10, 20],
            "predicted_sales": [8, 23],
        }
    )

    # Run the dashboard's daily error preparation function.
    result = prepare_daily_error_chart(predictions)

    # Actual minus predicted is positive for underprediction.
    assert result["error"].tolist() == [2, -3]

    # Absolute error removes the sign.
    assert result["absolute_error"].tolist() == [2, 3]


def test_filter_predictions_selects_requested_store_and_item():
    """Filtering should retain only matching store-item observations."""

    # Build a dataset containing multiple stores and items.
    predictions = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "store": [1, 1, 2],
            "item": [1, 2, 1],
            "actual_sales": [10, 15, 20],
            "predicted_sales": [11, 14, 18],
        }
    )

    # Request store 1 and item 2 only.
    result = filter_predictions(
        predictions,
        selected_store=1,
        selected_item=2,
    )

    # Only the matching store-item row should remain.
    assert len(result) == 1
    assert result.iloc[0]["store"] == 1
    assert result.iloc[0]["item"] == 2
