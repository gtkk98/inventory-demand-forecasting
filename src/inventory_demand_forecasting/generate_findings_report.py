
"""Generate a consolidated report from forecasting evaluation results."""

from pathlib import Path

import pandas as pd


# Resolve the project root from this Python file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# All input reports and the generated findings report live in reports/.
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "forecast_error_findings.md"


# Define the existing CSV reports that this module needs.
INPUT_FILES = {
    "summary": REPORTS_DIR / "xgboost_error_summary.csv",
    "weekday": REPORTS_DIR / "xgboost_error_by_weekday.csv",
    "month": REPORTS_DIR / "xgboost_error_by_month.csv",
    "store": REPORTS_DIR / "xgboost_error_by_store.csv",
    "item": REPORTS_DIR / "xgboost_error_by_item.csv",
    "store_item": REPORTS_DIR / "xgboost_error_by_store_item.csv",
    "comparison": REPORTS_DIR / "model_comparison.csv",
}


def load_report(path: Path) -> pd.DataFrame:
    """Load one CSV report and reject missing or empty files."""

    # A missing report means an earlier analysis step may not have run.
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}. "
            "Run the relevant analysis module first."
        )

    # Load the CSV into a DataFrame for analysis.
    df = pd.read_csv(path)

    # Empty reports cannot support meaningful findings.
    if df.empty:
        raise ValueError(f"Report contains no data: {path}")

    return df


def require_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    report_name: str,
) -> None:
    """Check that a report contains all columns needed for analysis."""

    # Calculate which expected columns are absent from the DataFrame.
    missing_columns = required_columns - set(df.columns)

    # Stop early with a useful message rather than failing later.
    if missing_columns:
        raise ValueError(
            f"{report_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )


def build_findings() -> str:
    """Build a Markdown report from the saved evaluation CSV files."""

    # Load each report using the common validation function.
    reports = {
        name: load_report(path)
        for name, path in INPUT_FILES.items()
    }

    # Give each DataFrame a readable variable name.
    summary = reports["summary"]
    weekday = reports["weekday"]
    month = reports["month"]
    store = reports["store"]
    item = reports["item"]
    store_item = reports["store_item"]
    comparison = reports["comparison"]

    # Validate the key fields used in the findings below.
    require_columns(
        summary,
        {"prediction_rows", "MAE", "RMSE", "WAPE_percent",
         "mean_error", "underprediction_rows",
         "overprediction_rows", "actual_sales_total",
         "predicted_sales_total"},
        "XGBoost summary",
    )

    require_columns(
        weekday,
        {"day_name", "MAE", "WAPE_percent"},
        "Weekday report",
    )

    require_columns(
        month,
        {"month_name", "MAE", "WAPE_percent"},
        "Monthly report",
    )

    require_columns(
        store,
        {"store", "MAE", "WAPE_percent"},
        "Store report",
    )

    require_columns(
        item,
        {"item", "MAE", "WAPE_percent"},
        "Item report",
    )

    require_columns(
        store_item,
        {"store", "item", "MAE", "WAPE_percent"},
        "Store-item report",
    )

    require_columns(
        comparison,
        {"model", "MAE", "RMSE", "WAPE_percent"},
        "Model comparison",
    )

    # There should be one overall summary row.
    if len(summary) != 1:
        raise ValueError(
            "XGBoost summary must contain exactly one overall row."
        )

    # Extract the single overall summary record.
    overall = summary.iloc[0]

    # Identify the best and worst groups using WAPE.
    # WAPE is a relative error measure, so lower values are better.
    best_weekday = weekday.loc[weekday["WAPE_percent"].idxmin()]
    worst_weekday = weekday.loc[weekday["WAPE_percent"].idxmax()]

    best_month = month.loc[month["WAPE_percent"].idxmin()]
    worst_month = month.loc[month["WAPE_percent"].idxmax()]

    worst_store = store.loc[store["WAPE_percent"].idxmax()]
    worst_item = item.loc[item["WAPE_percent"].idxmax()]
    worst_combination = store_item.loc[
        store_item["WAPE_percent"].idxmax()
    ]

    # Rank the models by WAPE so the report reflects the latest comparison.
    ranked_models = comparison.sort_values(
        "WAPE_percent",
        ascending=True,
    )

    best_model = ranked_models.iloc[0]

    # Calculate the difference between actual and predicted aggregate sales.
    total_sales_difference = (
        overall["predicted_sales_total"]
        - overall["actual_sales_total"]
    )

    # Format the findings as Markdown for easy reading in GitHub or editors.
    report_text = f"""# Inventory & Demand Forecasting System
## Forecast Error Analysis Findings

### 1. Evaluation overview

The XGBoost model was evaluated on **{int(overall["prediction_rows"]):,}**
predictions.

| Metric | Result |
|---|---:|
| MAE | {overall["MAE"]:.4f} |
| RMSE | {overall["RMSE"]:.4f} |
| WAPE | {overall["WAPE_percent"]:.4f}% |
| Mean error (actual - predicted) | {overall["mean_error"]:.4f} |
| Actual total sales | {overall["actual_sales_total"]:,.0f} |
| Predicted total sales | {overall["predicted_sales_total"]:,.2f} |
| Predicted minus actual sales | {total_sales_difference:,.2f} |

A positive mean error indicates a tendency to underpredict demand on average.
A negative mean error indicates a tendency to overpredict demand on average.

### 2. Forecasting error by time

- **Best weekday by WAPE:** {best_weekday["day_name"]}
  ({best_weekday["WAPE_percent"]:.4f}%).
- **Worst weekday by WAPE:** {worst_weekday["day_name"]}
  ({worst_weekday["WAPE_percent"]:.4f}%).
- **Best month by WAPE:** {best_month["month_name"]}
  ({best_month["WAPE_percent"]:.4f}%).
- **Worst month by WAPE:** {worst_month["month_name"]}
  ({worst_month["WAPE_percent"]:.4f}%).

These results describe the 2017 test period. They do not establish that
the same patterns will necessarily continue in future years.

### 3. Store and item analysis

- **Highest-WAPE store:** Store {int(worst_store["store"])}
  ({worst_store["WAPE_percent"]:.4f}%).
- **Highest-WAPE item:** Item {int(worst_item["item"])}
  ({worst_item["WAPE_percent"]:.4f}%).
- **Highest-WAPE store-item combination:** Store
  {int(worst_combination["store"])}, Item
  {int(worst_combination["item"])}
  ({worst_combination["WAPE_percent"]:.4f}%).

These groups are candidates for further investigation. High WAPE alone does
not prove that a product experienced stockouts or excess inventory.

### 4. Model comparison

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
{chr(10).join(
    f"| {row['model']} | {row['MAE']:.4f} | "
    f"{row['RMSE']:.4f} | {row['WAPE_percent']:.4f}% |"
    for _, row in ranked_models.iterrows()
)}

The best model by WAPE in the saved comparison is **{best_model["model"]}**.

### 5. Recommended next investigations

1. Investigate why Item {int(worst_item["item"])} has relatively high WAPE.
2. Examine demand patterns for Store {int(worst_store["store"])}.
3. Inspect daily errors and unusually large misses, not only average metrics.
4. Add stock availability, inventory levels, and supplier lead times before
   drawing conclusions about stockouts or replenishment performance.
5. Use a separate validation period for future model tuning, preserving the
   test period for an unbiased final evaluation.

### 6. Limitations

- The evaluation represents the available 2017 test period.
- The workflow evaluates rolling one-day-ahead predictions using historical
  sales as they become available.
- The dataset does not itself provide inventory on hand, supplier lead times,
  or stockout records.
- Forecast accuracy metrics do not directly measure inventory cost,
  service level, or lost sales.

### 7. Generated artifacts

- `model_comparison.csv`
- `xgboost_error_summary.csv`
- `xgboost_error_by_weekday.csv`
- `xgboost_error_by_month.csv`
- `xgboost_error_by_store.csv`
- `xgboost_error_by_item.csv`
- `xgboost_error_by_store_item.csv`
- `figures/`

"""

    # Return the report text so it can be saved or tested independently.
    return report_text


def main() -> None:
    """Generate and save the consolidated findings report."""

    # Ensure the output directory exists before writing the report.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build the report from the latest CSV files.
    report_text = build_findings()

    # Write UTF-8 Markdown so the report works in editors and GitHub.
    OUTPUT_PATH.write_text(report_text, encoding="utf-8")

    # Print the location to make the generated artifact easy to find.
    print("Forecast findings report generated successfully.")
    print(f"Report location: {OUTPUT_PATH}")


if __name__ == "__main__":
    # Run the report generator when this module is executed directly.
    main()
