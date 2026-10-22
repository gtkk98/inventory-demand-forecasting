
"""Automated tests for the demand forecasting API."""

from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

# Import the FastAPI application and its module for mocking.
from inventory_demand_forecasting.api import main as api_main

# Import the feature builder to test its real feature calculations.
from inventory_demand_forecasting.api.forecast_service import (
    FEATURE_COLUMNS,
    build_forecast_features,
)

# Create a test client without starting a separate Uvicorn server.
client = TestClient(api_main.app)


def test_forecast_endpoint_returns_prediction(monkeypatch):
    """A valid request should return the expected forecast response."""

    # Replace the real prediction service with a predictable test result.
    # This keeps the API contract test independent of model execution.
    def fake_forecast_sales(store, item, forecast_date):
        return {
            "store": store,
            "item": item,
            "date": forecast_date,
            "predicted_sales": 20.5,
            "model_name": "XGBoost",
        }

    # Redirect calls in main.py to our predictable test function.
    monkeypatch.setattr(
        api_main,
        "forecast_sales",
        fake_forecast_sales,
    )

    # Submit a valid forecast request to the API.
    response = client.post(
        "/forecast",
        json={
            "store": 1,
            "item": 1,
            "date": "2018-01-01",
        },
    )

    # Check the HTTP status and important response fields.
    assert response.status_code == 200

    body = response.json()
    assert body["store"] == 1
    assert body["item"] == 1
    assert body["date"] == "2018-01-01"
    assert body["predicted_sales"] == 20.5
    assert body["model_name"] == "XGBoost"


def test_forecast_rejects_invalid_store():
    """A store ID outside the allowed range should return HTTP 422."""

    response = client.post(
        "/forecast",
        json={
            "store": 11,
            "item": 1,
            "date": "2018-01-01",
        },
    )

    # Pydantic should reject the request before forecasting begins.
    assert response.status_code == 422


def test_forecast_rejects_invalid_item():
    """An item ID outside the allowed range should return HTTP 422."""

    response = client.post(
        "/forecast",
        json={
            "store": 1,
            "item": 51,
            "date": "2018-01-01",
        },
    )

    # The API should reject an unknown item ID.
    assert response.status_code == 422


def test_feature_builder_returns_expected_features():
    """The feature builder should produce one complete row of 14 features."""

    # Use a date immediately after the available historical dataset.
    features = build_forecast_features(
        store=1,
        item=1,
        forecast_date=date(2018, 1, 1),
    )

    # Check the shape and exact column order expected by the model.
    assert features.shape == (1, 14)
    assert features.columns.tolist() == FEATURE_COLUMNS

    # All generated features must be available for model inference.
    assert not features.isna().any().any()

    # The forecast row should contain the requested store and item.
    assert features.iloc[0]["store"] == 1
    assert features.iloc[0]["item"] == 1

    # The previous day's sales must be the final day's historical sales.
    assert features.iloc[0]["sales_lag_1"] == 23.0


def test_feature_builder_rejects_store_outside_range():
    """The feature builder should reject an invalid store ID."""

    try:
        build_forecast_features(
            store=11,
            item=1,
            forecast_date=date(2018, 1, 1),
        )
    except ValueError as exc:
        assert "Store must be between 1 and 10" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid store.")


def test_feature_builder_rejects_date_without_enough_history():
    """Dates with fewer than 30 previous observations should be rejected."""

    # January 1, 2013 is the first date in the dataset.
    # There is not enough earlier history to calculate the rolling features.
    try:
        build_forecast_features(
            store=1,
            item=1,
            forecast_date=date(2013, 1, 1),
        )
    except ValueError as exc:
        assert "At least 30 days" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError when insufficient history is available."
        )


def test_feature_builder_uses_only_sales_before_forecast_date():
    """The target date's actual sales must not leak into forecast features."""

    # This date has complete historical data available before it.
    features = build_forecast_features(
        store=1,
        item=1,
        forecast_date=date(2018, 1, 1),
    )

    # The most recent actual sales value before January 1, 2018 is 23.
    assert features.iloc[0]["sales_lag_1"] == 23.0

    # The requested target date's calendar features must be correct.
    assert features.iloc[0]["day_of_week"] == 0
    assert features.iloc[0]["month"] == 1
    assert features.iloc[0]["year"] == 2018
