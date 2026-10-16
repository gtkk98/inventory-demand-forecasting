from pathlib import Path

import matplotlib

# Use a non-interactive backend so charts can be saved from a terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
FIGURES_PATH = PROJECT_ROOT / "reports" / "figures"
REPORTS_PATH = PROJECT_ROOT / "reports"

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

def load_exploration_data()-> pd.DataFrame:
    """Load the dataset and prepare the date column for analysis."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    required_columns = {"date", "store", "item", "sales"}
    missing = required_columns - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    
    return df 

def create_summery(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate high-level dataset statistics."""
    summary = {
        "total_records": len(df),
        "unique_dates": df["date"].nunique(),
        "unique_stores": df["store"].nunique(),
        "unique_products": df["item"].nunique(),
        "total_sales": df["sales"].sum(),
        "average_sales_per_record": df["sales"].mean(),
        "median_sales_per_record": df["sales"].median(),
        "minimum_date": df["date"].min().date(),
        "maximum_date": df["date"].max().date(),
    }
    
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    
def analyze_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product sales across all stores for each date."""
    daily = (
        df.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    daily["day_of_week"] = daily["date"].dt.day_name()
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year

    return daily

def analyze_weekday_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean sales per store-item record across weekdays."""
    work = df[["date", "sales"]].copy()
    work["day_of_week"] = work["date"].dt.day_name()

    weekday = (
        work.groupby("day_of_week")["sales"]
        .agg(["mean", "median", "count"])
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )

    weekday.columns = [
        "day_of_week",
        "average_sales_per_record",
        "median_sales_per_record",
        "record_count",
    ]

    return weekday

def create_monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate total sales per calendar month."""
    work = daily.set_index("date")

    monthly = (
        work["sales"]
        .resample("MS")
        .sum()
        .rename("total_sales")
        .reset_index()
    )

    monthly["year_month"] = monthly["date"].dt.strftime("%Y-%m")

    return monthly

def plot_daily_sales(daily: pd.DataFrame) -> None:
    """Save a chart of total daily sales."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(daily["date"], daily["sales"], linewidth=0.7)
    ax.set_title("Total Daily Sales Across All Stores and Products")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "daily_sales.png", dpi=150)
    plt.close(fig)


def plot_monthly_sales(monthly: pd.DataFrame) -> None:
    """Save a chart of total monthly sales."""
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        monthly["date"],
        monthly["total_sales"],
        marker="o",
        markersize=2,
        linewidth=1.2,
    )
    ax.set_title("Total Monthly Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Sold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "monthly_sales.png", dpi=150)
    plt.close(fig)


def plot_weekday_patterns(weekday: pd.DataFrame) -> None:
    """Save a chart of average sales by weekday."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        weekday["day_of_week"],
        weekday["average_sales_per_record"],
    )
    ax.set_title("Average Sales per Store-Product Record by Weekday")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Units Sold per Record")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "weekday_sales.png", dpi=150)
    plt.close(fig)


def plot_store_sales(df: pd.DataFrame) -> None:
    """Save a comparison of total sales by store."""
    store_sales = (
        df.groupby("store")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(store_sales.index.astype(str), store_sales.values)
    ax.set_title("Total Sales by Store")
    ax.set_xlabel("Store ID")
    ax.set_ylabel("Total Units Sold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "store_sales.png", dpi=150)
    plt.close(fig)


def run_exploration() -> None:
    """Run the EDA pipeline and save summary tables and figures."""
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    df = load_exploration_data()

    summary = create_summery(df)
    daily = analyze_daily_sales(df)
    weekday = analyze_weekday_patterns(df)
    monthly = create_monthly_summary(daily)

    # Save reproducible outputs for reporting and later analysis.
    summary.to_csv(REPORTS_PATH / "dataset_summary.csv", index=False)
    daily.to_csv(REPORTS_PATH / "daily_sales_summary.csv", index=False)
    weekday.to_csv(REPORTS_PATH / "weekday_sales_summary.csv", index=False)
    monthly.to_csv(REPORTS_PATH / "monthly_sales_summary.csv", index=False)

    plot_daily_sales(daily)
    plot_monthly_sales(monthly)
    plot_weekday_patterns(weekday)
    plot_store_sales(df)

    print("\n=== DATASET SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== WEEKDAY SALES PATTERNS ===")
    print(weekday.to_string(index=False))

    print("\n=== TOP 5 STORES BY TOTAL SALES ===")
    print(
        df.groupby("store")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_string()
    )

    print("\n=== TOP 10 PRODUCTS BY TOTAL SALES ===")
    print(
        df.groupby("item")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    print(f"\nCharts saved in: {FIGURES_PATH}")
    print(f"Summary reports saved in: {REPORTS_PATH}")


def plot_item_sales(df: pd.DataFrame, item_id: int, store_id: int) -> None:
    """Plot historical daily sales for one product at one store."""
    selected = df.loc[
        (df["item"] == item_id) & (df["store"] == store_id)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No sales records found for item {item_id}, store {store_id}."
        )

    selected = selected.sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(selected["date"], selected["sales"], linewidth=0.9)

    ax.set_title(f"Daily Sales: Item {item_id}, Store {store_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = FIGURES_PATH / f"item_{item_id}_store_{store_id}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Product-level chart saved to: {output_path}")
    
    plot_item_sales(df, item_id=1, store_id=1)


if __name__ == "__main__":
    run_exploration()