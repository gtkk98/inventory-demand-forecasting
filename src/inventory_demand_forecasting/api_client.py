"""HTTP client for communicating with the forecasting FastAPI service."""

# Read environment variables for configurable API URLs.
import os

# Represent the requested forecast date.
from datetime import date

# Type JSON responses as dictionaries.
from typing import Any

# Send HTTP requests to the FastAPI server.
import httpx


# Use localhost unless a different API URL is configured.
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"

# Read the configured API URL and remove a trailing slash.
API_BASE_URL = os.getenv(
    "FORECAST_API_URL",
    DEFAULT_API_BASE_URL,
).rstrip("/")


def check_api_health(timeout: float = 5.0) -> dict[str, Any]:
    """Check whether the forecasting API is responding."""

    # Send a GET request to the API health endpoint.
    response = httpx.get(
        f"{API_BASE_URL}/health",
        timeout=timeout,
    )

    # Raise an exception if the server reports an HTTP error.
    response.raise_for_status()

    # Return the health information as a Python dictionary.
    return response.json()


def request_forecast(
    store: int,
    item: int,
    forecast_date: date,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Request a forecast for a particular store, item, and date."""

    # Prepare the JSON request expected by the /forecast endpoint.
    payload = {
        "store": store,
        "item": item,
        "date": forecast_date.isoformat(),
    }

    # Submit the forecasting request to FastAPI.
    response = httpx.post(
        f"{API_BASE_URL}/forecast",
        json=payload,
        timeout=timeout,
    )

    # Raise an exception for unsuccessful HTTP responses.
    response.raise_for_status()

    # Return the model's JSON prediction.
    return response.json()
