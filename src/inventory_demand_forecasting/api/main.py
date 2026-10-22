
"""FastAPI application entry point for demand forecasting."""

# Import FastAPI to create the REST API application.
from fastapi import FastAPI

# Import Path to inspect project files independently of the working directory.
from pathlib import Path

# Import timezone-aware date and time utilities for health responses.
from datetime import datetime, timezone


# Resolve the project root.
# main.py -> api -> inventory_demand_forecasting -> src -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Define the location of the existing trained XGBoost pipeline.
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"


# Create the FastAPI application and configure its API documentation.
app = FastAPI(
    title="Inventory & Demand Forecasting API",
    description=(
        "REST API for the Inventory & Demand Forecasting System. "
        "This initial version provides service health information. "
        "Forecasting endpoints will be added in subsequent issues."
    ),
    version="1.0.0",
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Provide a welcome message and links to API documentation."""

    # Return a small JSON response for anyone opening the API root.
    return {
        "message": "Inventory & Demand Forecasting API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str | bool]:
    """Report whether the API is running and the model artifact exists."""

    # Check the artifact's presence without loading the model into memory.
    model_exists = MODEL_PATH.is_file()

    # Use UTC so the timestamp does not depend on the server's local timezone.
    current_time = datetime.now(timezone.utc).isoformat()

    # Distinguish an available API from one whose model file is missing.
    return {
        "status": "ok" if model_exists else "degraded",
        "api_running": True,
        "model_artifact_exists": model_exists,
        "timestamp_utc": current_time,
    }
