
import pandas as pd
import pytest

from inventory_demand_forecasting.compare_models import compare_models


def make_metrics(
    model: str,
    mae: float,
    rmse: float,
    wape: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "MAE": mae,
                "RMSE": rmse,
                "WAPE_percent": wape,
            }
        ]
    )


def test_models_are_ranked_by_wape():
    previous_day = make_metrics(
        "Previous-day naive", 11.2, 15.4, 19.1
    )
    linear = make_metrics(
        "Linear Regression", 7.1, 9.2, 12.0
    )
    xgboost = make_metrics(
        "XGBoost", 6.1, 7.9, 10.3
    )

    result = compare_models(previous_day, linear, xgboost)

    assert result["model"].tolist() == [
        "XGBoost",
        "Linear Regression",
        "Previous-day naive",
    ]


def test_missing_metric_column_is_rejected():
    invalid = pd.DataFrame(
        [{"model": "Example", "MAE": 1.0, "RMSE": 2.0}]
    )

    with pytest.raises(ValueError, match="missing columns"):
        compare_models(invalid)


def test_missing_metric_value_is_rejected():
    invalid = make_metrics("Example", 1.0, 2.0, float("nan"))

    with pytest.raises(ValueError, match="missing values"):
        compare_models(invalid)


def test_negative_metric_is_rejected():
    invalid = make_metrics("Example", -1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="negative error metrics"):
        compare_models(invalid)


def test_duplicate_model_name_is_rejected():
    first = make_metrics("XGBoost", 6.0, 8.0, 10.0)
    second = make_metrics("XGBoost", 7.0, 9.0, 12.0)

    with pytest.raises(ValueError, match="Duplicate model names"):
        compare_models(first, second)


def test_no_metrics_tables_is_rejected():
    with pytest.raises(ValueError, match="At least one metrics table"):
        compare_models()
