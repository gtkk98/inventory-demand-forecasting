
from __future__ import annotations

import math

import numpy as np
import pandas as pd


SERVICE_LEVEL_Z = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
}

REQUIRED_SALES_COLUMNS = {"date", "store", "item", "sales"}
REQUIRED_INVENTORY_COLUMNS = {
    "store",
    "item",
    "current_stock",
    "lead_time_days",
}


def generate_inventory_recommendations(
    sales_history: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    service_level: float = 0.95,
    target_days_of_stock: int = 30,
) -> pd.DataFrame:
    """Calculate reorder recommendations for each store-item combination.

    sales_history must contain date, store, item and sales.
    inventory must contain store, item, current_stock and lead_time_days.
    """
    missing_sales = REQUIRED_SALES_COLUMNS - set(sales_history.columns)
    missing_inventory = REQUIRED_INVENTORY_COLUMNS - set(inventory.columns)

    if missing_sales:
        raise ValueError(
            f"Sales history is missing columns: {sorted(missing_sales)}"
        )

    if missing_inventory:
        raise ValueError(
            f"Inventory is missing columns: {sorted(missing_inventory)}"
        )

    if service_level not in SERVICE_LEVEL_Z:
        raise ValueError(
            f"service_level must be one of {list(SERVICE_LEVEL_Z)}"
        )

    if target_days_of_stock < 1:
        raise ValueError("target_days_of_stock must be at least 1")

    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "store",
                "item",
                "current_stock",
                "avg_daily_demand",
                "daily_demand_std",
                "safety_stock",
                "reorder_point",
                "target_stock",
                "recommended_order_quantity",
                "status",
            ]
        )

    history = sales_history.copy()
    stock = inventory.copy()

    history["date"] = pd.to_datetime(history["date"], errors="raise")

    for column in ("store", "item", "sales"):
        history[column] = pd.to_numeric(history[column], errors="raise")

    for column in ("store", "item", "current_stock", "lead_time_days"):
        stock[column] = pd.to_numeric(stock[column], errors="raise")

    if history[["store", "item", "sales"]].isna().any().any():
        raise ValueError("Sales history contains missing required values")

    if stock[list(REQUIRED_INVENTORY_COLUMNS)].isna().any().any():
        raise ValueError("Inventory contains missing required values")

    if (history["sales"] < 0).any():
        raise ValueError("Sales values cannot be negative")

    if (stock["current_stock"] < 0).any():
        raise ValueError("Current stock cannot be negative")

    if (stock["lead_time_days"] < 1).any():
        raise ValueError("Lead time must be at least 1 day")

    if (stock["lead_time_days"] % 1 != 0).any():
        raise ValueError("Lead time must use whole days")

    if (stock["current_stock"] % 1 != 0).any():
        raise ValueError("Current stock must use whole units")

    if stock.duplicated(["store", "item"]).any():
        raise ValueError("Inventory has duplicate store-item combinations")

    daily_sales = (
        history.groupby(["store", "item", "date"], as_index=False)["sales"]
        .sum()
    )

    demand_stats = (
        daily_sales.groupby(["store", "item"])["sales"]
        .agg(avg_daily_demand="mean", daily_demand_std="std")
        .reset_index()
    )

    result = stock.merge(
        demand_stats,
        on=["store", "item"],
        how="left",
        validate="one_to_one",
    )

    if result["avg_daily_demand"].isna().any():
        missing_pairs = result.loc[
            result["avg_daily_demand"].isna(), ["store", "item"]
        ].to_dict("records")
        raise ValueError(f"No sales history for inventory items: {missing_pairs}")

    result["daily_demand_std"] = result["daily_demand_std"].fillna(0.0)

    z_score = SERVICE_LEVEL_Z[service_level]

    result["safety_stock"] = (
        z_score
        * result["daily_demand_std"]
        * np.sqrt(result["lead_time_days"])
    ).apply(math.ceil)

    result["reorder_point"] = (
        result["avg_daily_demand"] * result["lead_time_days"]
        + result["safety_stock"]
    ).apply(math.ceil)

    result["target_stock"] = (
        result["avg_daily_demand"]
        * (result["lead_time_days"] + target_days_of_stock)
        + result["safety_stock"]
    ).apply(math.ceil)

    result["recommended_order_quantity"] = (
        result["target_stock"] - result["current_stock"]
    ).clip(lower=0).astype(int)

    result["status"] = np.where(
        result["current_stock"] <= result["reorder_point"],
        "REORDER",
        "MONITOR",
    )

    columns = [
        "store",
        "item",
        "current_stock",
        "avg_daily_demand",
        "daily_demand_std",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order_quantity",
        "status",
    ]

    return result[columns].sort_values(
        ["status", "store", "item"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
