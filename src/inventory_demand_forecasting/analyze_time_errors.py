"""Analyze XGBoost forecast errors across time periods."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "reports" / "xgboost_error_analysis.csv"
REPORTS_DIR = PROJECT_ROOT / 'reports'

WEEKDAY_REPORT = REPORTS_DIR / "xgboost_error_by_weekday.csv"
MONTH_REPORT = REPORTS_DIR / "xgboost_error_by_month.csv"
DAILY_REPORT = REPORTS_DIR / "xgboost_daily_error_trend.csv"

REQUIRED_COLUMNS = [
    "date",
    "actual_sales",
    "predicted_sales",
    "error",
    "absolute_error",
    "squared_error",
]

def load_error_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the detailed error analysis report."""
    
    if not path.exists():
        raise FileNotFoundError(
            f"Error analysis report not found: {path}. "
            "Run analyze_errors first."
        )
        
    df = pd.read_csv(path)
    
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing:
        raise ValueError(
            f"Error report is missing columns: {sorted(missing)}"
        )
    
    if df.empty:
        raise ValueError("Error analysis report contains no rows.")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    numeric_columns = [
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
    
    # Verify that the saved error calculations are consistent.
    expected_error = df["actual_sales"] - df["predicted_sales"]
    
    if not np.allclose(df["error"], expected_error):
        raise ValueError("Saved error values do not match actual - predicted.")
    
    if not np.allclose(df["absolute_error"], expected_error.abs()):
        raise ValueError("Saved absolute errors are inconsistent.")
    
    return df

def summarize_group(group: pd.DataFrame) -> pd.Series:
    """Calculate forecasting metrics for one time group."""
    
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

def build_time_reports(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build weekday, monthly, and daily error reports."""

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_name"] = data["date"].dt.day_name()
    data["month"] = data["date"].dt.month
    data["month_name"] = data["date"].dt.month_name()

    weekday_report = (
        data.groupby(["day_of_week", "day_name"], sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    month_report = (
        data.groupby(["month", "month_name"], sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    daily_report = (
        data.groupby("date", sort=True)
        .apply(summarize_group, include_groups=False)
        .reset_index()
    )

    return weekday_report, month_report, daily_report


def main() -> None:
    """Run temporal error analysis and save reports."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_error_data()

    weekday_report, month_report, daily_report = build_time_reports(data)

    weekday_report.to_csv(WEEKDAY_REPORT, index=False)
    month_report.to_csv(MONTH_REPORT, index=False)
    daily_report.to_csv(DAILY_REPORT, index=False)

    print("\n=== FORECAST ERRORS BY DAY OF WEEK ===")
    print(
        weekday_report[
            ["day_name", "prediction_rows", "MAE", "RMSE", "WAPE_percent", "mean_error"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=== FORECAST ERRORS BY MONTH ===")
    print(
        month_report[
            ["month_name", "prediction_rows", "MAE", "RMSE", "WAPE_percent", "mean_error"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    worst_weekday = weekday_report.loc[weekday_report["WAPE_percent"].idxmax()]
    best_weekday = weekday_report.loc[weekday_report["WAPE_percent"].idxmin()]

    worst_month = month_report.loc[month_report["WAPE_percent"].idxmax()]
    best_month = month_report.loc[month_report["WAPE_percent"].idxmin()]

    print("\n=== KEY FINDINGS ===")
    print(
        f"Highest weekday WAPE: {worst_weekday['day_name']} "
        f"({worst_weekday['WAPE_percent']:.4f}%)"
    )
    print(
        f"Lowest weekday WAPE: {best_weekday['day_name']} "
        f"({best_weekday['WAPE_percent']:.4f}%)"
    )
    print(
        f"Highest monthly WAPE: {worst_month['month_name']} "
        f"({worst_month['WAPE_percent']:.4f}%)"
    )
    print(
        f"Lowest monthly WAPE: {best_month['month_name']} "
        f"({best_month['WAPE_percent']:.4f}%)"
    )

    print(f"\nWeekday report: {WEEKDAY_REPORT}")
    print(f"Monthly report:  {MONTH_REPORT}")
    print(f"Daily report:    {DAILY_REPORT}")


if __name__ == "__main__":
    main()
