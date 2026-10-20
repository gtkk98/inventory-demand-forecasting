import pandas as pd
import pytest

from inventory_demand_forecasting.lag_features import (
    add_lag_features
)

def test_lad_features_use_pervious_sales_within_each_group():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=9).tolist() 
            + pd.date_range("2026-01-01", periods=9).tolist(),
            "store": [1] * 9 + [2] * 9,
            "item": [1] * 18,
            "sales": list(range(10, 19)) + list(range(100, 109)),
        }
    )
    
    result = add_lag_features(df)
    
    store_1 = result[result["store"] == 1].reset_index(drop=True)
    store_2 = result[result["store"] == 2].reset_index(drop=True)
    
    assert pd.isna(store_1.loc[0, "sales_lag_1"])
    assert store_1.loc[1, "sales_lag_1"] == 10
    assert store_1.loc[7, "sales_lag_7"] == 10
    
    # Store 2 must use its own history, never Store 1's sales.
    assert pd.isna(store_2.loc[0, "sales_lag_1"])
    assert store_2.loc[1, "sales_lag_1"] == 100
    assert store_2.loc[7, "sales_lag_7"] == 100
    
def test_lag_features_do_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "store": [1, 1, 1],
            "item": [1, 1, 1],
            "sales": [10, 20, 30],
        }
    )
    
    original_columns = df.columns.tolist()
    
    result = add_lag_features(df)
    
    assert df.columns.tolist() == original_columns
    assert "sales_lag_1" in result.columns
    assert "sales_lag_7" in result.columns
    
def test_duplicate_store_item_date_raises_error():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01"]
            ),
            "store": [1, 1],
            "item": [1, 1],
            "sales": [10, 15],
        }
    )
    
    with pytest.raises(ValueError, match="Duplicate"):
        add_lag_features(df)

def test_missing_required_column_raises_error():
    df = pd.DataFrame(
        {
           "date": pd.to_datetime(["2026-01-01"]),
            "store": [1],
            "sales": [10], 
        }
    )
    
    with pytest.raises(ValueError, match="missing required columns"):
        add_lag_features(df)