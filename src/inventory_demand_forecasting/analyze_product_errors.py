
"""Analyze XGBoost forecasting errors by store and item."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "reports" / "xgboost_error_analysis.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

STORE_REPORT = REPORTS_DIR / "xgboost_error_by_store.csv"
ITEM_REPORT = REPORTS_DIR / "xgboost_error_by_item.csv"
STORE_ITEM_REPORT = REPORTS_DIR / "xgboost_error_by_store_item.csv"
WORST_COMBINATIONS_REPORT = (
    REPORTS_DIR / "xgboost_worst_store_item_combinations.csv"
)


REQUIRED_COLUMNS = [
    "date",
    "store",
    "item",
    "actual_sales",
    "predicted_sales",
    "error",
    "absolute_error",
    "squared_error",
]


def load_error_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the detailed forecasting error report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Error report not found: {path}. "
            "Run analyze_errors first."
        )

    df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Error report is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Error report contains no rows.")

    numeric_columns = [
        "store",
        "item",
        "actual_sales",
        "predicted_sales",
        "error",
        "absolute_error",
        "squared_error",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Column '{column}' contains non-finite values."
            )

    expected_error = df["actual_sales"] - df["predicted_sales"]

    if not np.allclose(df["error"], expected_error):
        raise ValueError("Saved forecast errors are inconsistent.")

    if df.duplicated(["date", "store", "item"]).any():
        raise ValueError("Duplicate date/store/item rows detected.")

    return df


def summarize_group(group: pd.DataFrame) -> pd.Series:
    """Calculate metrics for one store, item, or store-item group."""

    actual_total = group["actual_sales"].sum()

    wape = (
        group["absolute_error"].sum() / actual_total * 100
        if actual_total != 0
        else np.nan
    )

    return pd.Series(
        {
            "prediction_rows": len(group),
            "actual_sales_total": actual_total,
            "predicted_sales_total": group["predicted_sales"].sum(),
            "MAE": group["absolute_error"].mean(),
            "RMSE": np.sqrt(group["squared_error"].mean()),
            "WAPE_percent": wape,
            "mean_error": group["error"].mean(),
        }
    )


def build_product_reports(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create store-level, item-level, and store-item reports."""

    store_report = (
        df.groupby("store", sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
        .sort_values("WAPE_percent", ascending=False)
    )

    item_report = (
        df.groupby("item", sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
        .sort_values("WAPE_percent", ascending=False)
    )

    store_item_report = (
        df.groupby(["store", "item"], sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
        .sort_values("WAPE_percent", ascending=False)
    )

    return store_report, item_report, store_item_report


def main() -> None:
    """Run product-level error analysis and save reports."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_error_data()

    store_report, item_report, store_item_report = (
        build_product_reports(data)
    )

    store_report.to_csv(STORE_REPORT, index=False)
    item_report.to_csv(ITEM_REPORT, index=False)
    store_item_report.to_csv(STORE_ITEM_REPORT, index=False)

    # The ten combinations with the highest relative error.
    worst_combinations = store_item_report.head(10)
    worst_combinations.to_csv(WORST_COMBINATIONS_REPORT, index=False)

    display_columns = [
        "store",
        "item",
        "prediction_rows",
        "MAE",
        "RMSE",
        "WAPE_percent",
        "mean_error",
    ]

    print("\n=== STORE PERFORMANCE: HIGHEST WAPE FIRST ===")
    print(
        store_report[
            ["store", "prediction_rows", "MAE", "RMSE",
             "WAPE_percent", "mean_error"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=== ITEM PERFORMANCE: HIGHEST WAPE FIRST ===")
    print(
        item_report[
            ["item", "prediction_rows", "MAE", "RMSE",
             "WAPE_percent", "mean_error"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=== TEN STORE-ITEM COMBINATIONS WITH HIGHEST WAPE ===")
    print(
        worst_combinations[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nStore report: {STORE_REPORT}")
    print(f"Item report: {ITEM_REPORT}")
    print(f"Store-item report: {STORE_ITEM_REPORT}")
    print(f"Top 10 report: {WORST_COMBINATIONS_REPORT}")


if __name__ == "__main__":
    main()
