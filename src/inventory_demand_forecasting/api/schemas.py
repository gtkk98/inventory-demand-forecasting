"""Pydantic schemas for validating forecasting API requests and responses."""

# Alias the date type to avoid a naming collision with the `date` fields.
from datetime import date as DateType

# Import Pydantic's schema base class and field-validation helper.
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Validate incoming demand-forecast requests."""

    # Only the 10 stores present in our dataset are supported.
    store: int = Field(ge=1, le=10, description="Store ID, from 1 to 10.")

    # Only the 50 items present in our dataset are supported.
    item: int = Field(ge=1, le=50, description="Item ID, from 1 to 50.")

    # Validate the requested forecast date.
    date: DateType = Field(
        description="Date for which demand should be forecast.",
        examples=["2017-12-31"],
    )


class ForecastResponse(BaseModel):
    """Define the structure of a successful forecast response."""

    # Identify the store and item.
    store: int = Field(description="Store ID.")
    item: int = Field(description="Item ID.")

    # Return the date associated with the prediction.
    date: DateType = Field(description="Forecast date.")

    # Demand cannot be negative; decimal predictions are allowed.
    predicted_sales: float = Field(
        ge=0,
        description="Predicted units of demand.",
    )

    # Identify the forecasting model used.
    model_name: str = Field(
        description="Name of the forecasting model.",
        examples=["XGBoost"],
    )
