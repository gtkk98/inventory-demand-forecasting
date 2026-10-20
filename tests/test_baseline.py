
import numpy as np
import pandas as pd
import pytest

from inventory_demand_forecasting.baseline import (
    create_baseline_predictions,
)
from inventory_demand_forecasting.evaluate import calculate_metrics


def test_baseline_predictions_use_correct_lags():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-08", "2017-01-09"]),
            "store": [1, 1],
            "item": [1, 1],
            "sales": [20, 25],
            "sales_lag_1": [18, 20],
            "sales_lag_7": [15, 18],
        }
    )

    result = create_baseline_predictions(df)

    assert result["naive_lag_1_prediction"].tolist() == [18, 20]
    assert result["seasonal_naive_lag_7_prediction"].tolist() == [15, 18]


def test_metrics_calculate_expected_values():
    actual = pd.Series([10, 20, 30])
    predicted = pd.Series([12, 18, 33])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["MAE"] == pytest.approx(7 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(17 / 3))
    assert metrics["WAPE_percent"] == pytest.approx(7 / 60 * 100)


def test_metrics_reject_different_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_metrics(
            pd.Series([10, 20]),
            pd.Series([10]),
        )


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError, match="empty"):
        calculate_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))


def test_baseline_rejects_missing_lag_values():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01"]),
            "store": [1],
            "item": [1],
            "sales": [20],
            "sales_lag_1": [np.nan],
            "sales_lag_7": [15],
        }
    )

    with pytest.raises(ValueError, match="Missing"):
        create_baseline_predictions(df)
