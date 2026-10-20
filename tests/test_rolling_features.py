
import pandas as pd
import pytest

from inventory_demand_forecasting.rolling_features import (
    add_rolling_features,
)


def test_rolling_features_use_only_previous_sales():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=9),
            "store": [1] * 9,
            "item": [1] * 9,
            "sales": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        }
    )

    result = add_rolling_features(df)

    # January 8 uses sales from January 1 through January 7.
    jan_8 = result.loc[result["date"] == "2026-01-08"].iloc[0]

    assert jan_8["sales_rolling_mean_7"] == pytest.approx(40.0)
    assert jan_8["sales_rolling_mean_30"] != jan_8["sales_rolling_mean_30"]
    assert jan_8["sales_rolling_std_7"] == pytest.approx(
        pd.Series([10, 20, 30, 40, 50, 60, 70]).std()
    )


def test_rolling_features_are_separate_for_each_store():
    dates = pd.date_range("2026-01-01", periods=8)

    df = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "store": [1] * 8 + [2] * 8,
            "item": [1] * 16,
            "sales": list(range(1, 9)) + list(range(101, 109)),
        }
    )

    result = add_rolling_features(df)

    store_1_jan_8 = result[
        (result["store"] == 1)
        & (result["date"] == "2026-01-08")
    ].iloc[0]

    store_2_jan_8 = result[
        (result["store"] == 2)
        & (result["date"] == "2026-01-08")
    ].iloc[0]

    assert store_1_jan_8["sales_rolling_mean_7"] == pytest.approx(4.0)
    assert store_2_jan_8["sales_rolling_mean_7"] == pytest.approx(104.0)


def test_rolling_features_preserve_original_dataframe():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "store": [1, 1, 1],
            "item": [1, 1, 1],
            "sales": [10, 20, 30],
        }
    )

    original_columns = df.columns.tolist()

    result = add_rolling_features(df)

    assert df.columns.tolist() == original_columns
    assert "sales_rolling_mean_7" in result.columns
    assert "sales_rolling_mean_30" in result.columns
    assert "sales_rolling_std_7" in result.columns


def test_duplicate_date_store_item_raises_error():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01"]
            ),
            "store": [1, 1],
            "item": [1, 1],
            "sales": [10, 20],
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        add_rolling_features(df)


def test_missing_required_column_raises_error():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "store": [1],
            "sales": [10],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        add_rolling_features(df)
