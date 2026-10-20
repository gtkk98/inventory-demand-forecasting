
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from inventory_demand_forecasting.train_xgboost import validate_dataset
from inventory_demand_forecasting.xgboost_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_xgboost_pipeline,
)


def make_valid_dataset() -> pd.DataFrame:
    """Create a small valid dataset for testing."""

    data = {
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "store": [1, 1],
        "item": [1, 2],
        "sales": [20, 30],
        "day_of_week": [2, 3],
        "day_of_month": [1, 2],
        "month": [1, 1],
        "year": [2025, 2025],
        "quarter": [1, 1],
        "week_of_year": [1, 1],
        "is_weekend": [0, 0],
        "sales_lag_1": [18, 25],
        "sales_lag_7": [16, 22],
        "sales_rolling_mean_7": [17.0, 23.0],
        "sales_rolling_mean_30": [15.0, 21.0],
        "sales_rolling_std_7": [2.0, 3.0],
    }

    return pd.DataFrame(data)


def test_pipeline_contains_preprocessor_and_xgboost():
    pipeline = build_xgboost_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert isinstance(
        pipeline.named_steps["model"],
        XGBRegressor,
    )


def test_feature_columns_include_expected_features():
    assert "store" in FEATURE_COLUMNS
    assert "item" in FEATURE_COLUMNS
    assert "sales_lag_1" in FEATURE_COLUMNS
    assert "sales_lag_7" in FEATURE_COLUMNS
    assert "sales_rolling_mean_7" in FEATURE_COLUMNS
    assert TARGET_COLUMN == "sales"


def test_valid_dataset_passes_validation():
    df = make_valid_dataset()

    validate_dataset(df, "Test dataset")


def test_missing_feature_is_rejected():
    df = make_valid_dataset().drop(columns=["sales_lag_1"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(df, "Test dataset")


def test_missing_feature_value_is_rejected():
    df = make_valid_dataset()
    df.loc[0, "sales_lag_1"] = np.nan

    with pytest.raises(ValueError, match="missing feature values"):
        validate_dataset(df, "Test dataset")


def test_missing_target_value_is_rejected():
    df = make_valid_dataset()
    df.loc[0, "sales"] = np.nan

    with pytest.raises(ValueError, match="missing target values"):
        validate_dataset(df, "Test dataset")


def test_missing_date_is_rejected():
    df = make_valid_dataset()
    df.loc[0, "date"] = pd.NaT

    with pytest.raises(ValueError, match="missing dates"):
        validate_dataset(df, "Test dataset")


def test_empty_dataset_is_rejected():
    df = make_valid_dataset().iloc[0:0]

    with pytest.raises(ValueError, match="is empty"):
        validate_dataset(df, "Test dataset")
