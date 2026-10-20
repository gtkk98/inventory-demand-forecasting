
import numpy as np
import pandas as pd


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """Calculate common regression and forecasting metrics."""

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(predicted_values).all()
    ):
        raise ValueError(
            "Actual and predicted values must be finite numbers."
        )

    errors = actual_values - predicted_values

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    total_actual = float(np.sum(np.abs(actual_values)))

    wape = (
        float(np.sum(np.abs(errors)) / total_actual * 100)
        if total_actual != 0
        else float("nan")
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }
