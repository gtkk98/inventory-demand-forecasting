"""Tests for the forecasting API request and response schemas."""

# Import the date type to check date parsing.
from datetime import date

# Import pytest for parameterized tests and expected exceptions.
import pytest

# Import Pydantic's validation exception.
from pydantic import ValidationError

# Import the schemas being tested.
from inventory_demand_forecasting.api.schemas import (
    ForecastRequest,
    ForecastResponse,
)


def test_valid_forecast_request():
    """Accept valid store, item, and date values."""

    # Build a valid request using values within the dataset's ranges.
    request = ForecastRequest(
        store=1,
        item=1,
        date="2017-12-31",
    )

    # Verify the string date was parsed into a Python date object.
    assert request.date == date(2017, 12, 31)

    # Verify the store and item IDs were preserved.
    assert request.store == 1
    assert request.item == 1


@pytest.mark.parametrize("store", [0, 11])
def test_invalid_store_id_is_rejected(store):
    """Reject store IDs outside the supported range."""

    # Pydantic should reject store IDs below 1 or above 10.
    with pytest.raises(ValidationError):
        ForecastRequest(
            store=store,
            item=1,
            date="2017-12-31",
        )


@pytest.mark.parametrize("item", [0, 51])
def test_invalid_item_id_is_rejected(item):
    """Reject item IDs outside the supported range."""

    # Pydantic should reject item IDs below 1 or above 50.
    with pytest.raises(ValidationError):
        ForecastRequest(
            store=1,
            item=item,
            date="2017-12-31",
        )


def test_invalid_date_is_rejected():
    """Reject an impossible calendar date."""

    # February 30 does not exist and must fail validation.
    with pytest.raises(ValidationError):
        ForecastRequest(
            store=1,
            item=1,
            date="2017-02-30",
        )


def test_valid_forecast_response():
    """Accept a correctly structured forecast response."""

    # Create a representative response from a forecasting model.
    response = ForecastResponse(
        store=1,
        item=1,
        date="2017-12-31",
        predicted_sales=52.5,
        model_name="XGBoost",
    )

    # Check the prediction value and parsed date.
    assert response.predicted_sales == 52.5
    assert response.date == date(2017, 12, 31)


def test_negative_prediction_is_rejected():
    """Reject negative predicted demand."""

    # Negative sales predictions are invalid for this application.
    with pytest.raises(ValidationError):
        ForecastResponse(
            store=1,
            item=1,
            date="2017-12-31",
            predicted_sales=-5,
            model_name="XGBoost",
        )
