import pandas as pd
import pytest

from inventory_demand_forecasting.calendar_features import (
    add_calendar_features,
)

def test_calendar_features_for_known_dates():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-10-19", "2026-10-24", "2026-10-25"]
            ),
            "store": [1, 1, 1],
            "item": [1, 1, 1],
            "sales": [10, 20, 30],
        }
    )
    
    result = add_calendar_features(df)
    
    # Monday = 0, Saturday = 5, Sunday = 6.
    assert result["day_of_week"].tolist() == [0, 5, 6]
    assert result["day_name"].tolist() == [
        "Monday",
        "Saturday",
        "Sunday",
    ]
    
    assert result["is_weekend"].tolist() == [0, 1, 1]
    assert result["month"].tolist() == [10, 10, 10]
    assert result["quarter"].tolist() == [4, 4, 4]
    assert result["day_of_month"].tolist() == [19, 24, 25]
    
def test_calendar_features_preserve_original_columns():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"]),
            "store": [1],
            "item": [1],
            "sales": [15],
        }
    )

    result = add_calendar_features(df)

    assert result["store"].tolist() == [1]
    assert result["item"].tolist() == [1]
    assert result["sales"].tolist() == [15]

    # The input DataFrame must remain unchanged.
    assert list(df.columns) == [
        "date",
        "store",
        "item",
        "sales",
    ]


def test_missing_date_column_raises_error():
    df = pd.DataFrame({"sales": [10]})

    with pytest.raises(ValueError, match="date"):
        add_calendar_features(df)


def test_missing_date_values_raise_error():
    df = pd.DataFrame(
        {
            "date": [pd.NaT],
            "store": [1],
            "item": [1],
            "sales": [10],
        }
    )

    with pytest.raises(ValueError, match="missing"):
        add_calendar_features(df)
