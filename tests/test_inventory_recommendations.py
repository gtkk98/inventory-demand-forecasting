
import pandas as pd
import pytest

from inventory.recommendations import generate_inventory_recommendations


@pytest.fixture
def sales_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "store": [1, 1, 1, 1, 1],
            "item": [1, 1, 1, 1, 1],
            "sales": [10, 12, 8, 10, 10],
        }
    )


@pytest.fixture
def inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store": [1],
            "item": [1],
            "current_stock": [20],
            "lead_time_days": [2],
        }
    )


def test_generates_recommendation_for_low_stock(sales_history, inventory):
    result = generate_inventory_recommendations(sales_history, inventory)

    assert len(result) == 1
    assert result.loc[0, "avg_daily_demand"] == pytest.approx(10.0)
    assert result.loc[0, "safety_stock"] == 4
    assert result.loc[0, "reorder_point"] == 24
    assert result.loc[0, "target_stock"] == 324
    assert result.loc[0, "recommended_order_quantity"] == 304
    assert result.loc[0, "status"] == "REORDER"


def test_sufficient_stock_is_marked_monitor(sales_history, inventory):
    inventory["current_stock"] = 1000

    result = generate_inventory_recommendations(sales_history, inventory)

    assert result.loc[0, "status"] == "MONITOR"
    assert result.loc[0, "recommended_order_quantity"] == 0


def test_sales_for_same_day_are_aggregated(sales_history, inventory):
    extra_row = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")],
            "store": [1],
            "item": [1],
            "sales": [5],
        }
    )
    history = pd.concat([sales_history, extra_row], ignore_index=True)

    result = generate_inventory_recommendations(history, inventory)

    assert result.loc[0, "avg_daily_demand"] == pytest.approx(11.0)


def test_missing_sales_column_raises_error(sales_history, inventory):
    history = sales_history.drop(columns=["sales"])

    with pytest.raises(ValueError, match="Sales history is missing columns"):
        generate_inventory_recommendations(history, inventory)


def test_missing_inventory_column_raises_error(sales_history, inventory):
    stock = inventory.drop(columns=["current_stock"])

    with pytest.raises(ValueError, match="Inventory is missing columns"):
        generate_inventory_recommendations(sales_history, stock)


def test_inventory_without_sales_history_raises_error(sales_history, inventory):
    inventory["item"] = 2

    with pytest.raises(ValueError, match="No sales history"):
        generate_inventory_recommendations(sales_history, inventory)


def test_negative_sales_are_rejected(sales_history, inventory):
    sales_history.loc[0, "sales"] = -1

    with pytest.raises(ValueError, match="Sales values cannot be negative"):
        generate_inventory_recommendations(sales_history, inventory)


def test_zero_lead_time_is_rejected(sales_history, inventory):
    inventory["lead_time_days"] = 0

    with pytest.raises(ValueError, match="Lead time must be at least 1 day"):
        generate_inventory_recommendations(sales_history, inventory)


def test_unsupported_service_level_is_rejected(sales_history, inventory):
    with pytest.raises(ValueError, match="service_level must be one of"):
        generate_inventory_recommendations(
            sales_history,
            inventory,
            service_level=0.93,
        )


def test_empty_inventory_returns_empty_result(sales_history):
    empty_inventory = pd.DataFrame(
        columns=["store", "item", "current_stock", "lead_time_days"]
    )

    result = generate_inventory_recommendations(
        sales_history,
        empty_inventory,
    )

    assert result.empty
    assert "recommended_order_quantity" in result.columns
