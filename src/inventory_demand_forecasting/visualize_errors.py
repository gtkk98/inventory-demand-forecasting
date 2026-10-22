
"""Generate visualizations for XGBoost forecast error analysis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ERRORS_PATH = REPORTS_DIR / "xgboost_error_analysis.csv"
WEEKDAY_PATH = REPORTS_DIR / "xgboost_error_by_weekday.csv"
MONTH_PATH = REPORTS_DIR / "xgboost_error_by_month.csv"
DAILY_PATH = REPORTS_DIR / "xgboost_daily_error_trend.csv"
STORE_PATH = REPORTS_DIR / "xgboost_error_by_store.csv"
ITEM_PATH = REPORTS_DIR / "xgboost_error_by_item.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a report and ensure it exists and contains data."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}. "
            "Run the corresponding analysis module first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Report contains no data: {path}")

    return df


def save_figure(filename: str) -> None:
    """Save the current chart and close its figure."""

    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_error_distribution(errors: pd.DataFrame) -> None:
    """Visualize the distribution of signed forecasting errors."""

    plt.figure(figsize=(10, 6))

    plt.hist(errors["error"], bins=60, edgecolor="black")

    plt.axvline(
        0,
        linestyle="--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.title("XGBoost Forecast Error Distribution")
    plt.xlabel("Error (actual sales - predicted sales)")
    plt.ylabel("Number of predictions")
    plt.legend()

    save_figure("error_distribution.png")


def plot_weekday_wape(weekday: pd.DataFrame) -> None:
    """Compare relative forecasting error across weekdays."""

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday["day_name"] = pd.Categorical(
        weekday["day_name"],
        categories=weekday_order,
        ordered=True,
    )

    weekday = weekday.sort_values("day_name")

    plt.figure(figsize=(10, 6))

    plt.bar(
        weekday["day_name"].astype(str),
        weekday["WAPE_percent"],
    )

    plt.title("XGBoost WAPE by Day of Week")
    plt.xlabel("Day of week")
    plt.ylabel("WAPE (%)")
    plt.xticks(rotation=30)

    save_figure("wape_by_weekday.png")


def plot_monthly_wape(monthly: pd.DataFrame) -> None:
    """Compare relative forecasting error across months."""

    monthly = monthly.sort_values("month")

    plt.figure(figsize=(10, 6))

    plt.bar(
        monthly["month_name"],
        monthly["WAPE_percent"],
    )

    plt.title("XGBoost WAPE by Month")
    plt.xlabel("Month")
    plt.ylabel("WAPE (%)")
    plt.xticks(rotation=45)

    save_figure("wape_by_month.png")


def plot_daily_sales(daily: pd.DataFrame) -> None:
    """Compare daily aggregate actual and predicted sales."""

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    plt.figure(figsize=(12, 6))

    plt.plot(
        daily["date"],
        daily["actual_sales_total"],
        label="Actual sales",
        linewidth=1.2,
    )

    plt.plot(
        daily["date"],
        daily["predicted_sales_total"],
        label="Predicted sales",
        linewidth=1.2,
    )

    plt.title("Daily Actual vs Predicted Sales — 2017")
    plt.xlabel("Date")
    plt.ylabel("Total sales across all stores and items")
    plt.legend()

    save_figure("daily_actual_vs_predicted.png")


def plot_store_wape(stores: pd.DataFrame) -> None:
    """Compare relative forecasting error by store."""

    stores = stores.sort_values("WAPE_percent", ascending=True)

    plt.figure(figsize=(9, 6))

    plt.barh(
        stores["store"].astype(str),
        stores["WAPE_percent"],
    )

    plt.title("XGBoost WAPE by Store")
    plt.xlabel("WAPE (%)")
    plt.ylabel("Store")

    save_figure("wape_by_store.png")


def plot_item_wape(items: pd.DataFrame) -> None:
    """Show the ten items with the highest relative error."""

    worst_items = items.nlargest(10, "WAPE_percent")
    worst_items = worst_items.sort_values("WAPE_percent")

    plt.figure(figsize=(9, 6))

    plt.barh(
        worst_items["item"].astype(str),
        worst_items["WAPE_percent"],
    )

    plt.title("10 Items with Highest XGBoost WAPE")
    plt.xlabel("WAPE (%)")
    plt.ylabel("Item")

    save_figure("top_10_items_by_wape.png")


def main() -> None:
    """Generate all forecast error visualizations."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    errors = load_csv(ERRORS_PATH)
    weekday = load_csv(WEEKDAY_PATH)
    monthly = load_csv(MONTH_PATH)
    daily = load_csv(DAILY_PATH)
    stores = load_csv(STORE_PATH)
    items = load_csv(ITEM_PATH)

    required_error_columns = {
        "error",
        "actual_sales",
        "predicted_sales",
    }

    if not required_error_columns.issubset(errors.columns):
        raise ValueError("Error details report is missing required columns.")

    plot_error_distribution(errors)
    plot_weekday_wape(weekday)
    plot_monthly_wape(monthly)
    plot_daily_sales(daily)
    plot_store_wape(stores)
    plot_item_wape(items)

    print("\nAll six forecast error charts were generated successfully.")
    print(f"Chart directory: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
