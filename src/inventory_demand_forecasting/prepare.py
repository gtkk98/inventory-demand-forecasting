
from pathlib import Path

import pandas as pd

from inventory_demand_forecasting.cleaning import (
    load_raw_data,
    inspect_data,
    inspect_time_series_completeness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

PROCESSED_DATA_PATH = PROCESSED_DIR / "sales_clean.csv"


def review_sales_distribution(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize sales distribution and identify potential outliers."""

    sales = df["sales"]

    quantiles = sales.quantile(
        [0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    )

    distribution = pd.DataFrame(
        {
            "metric": [
                "minimum",
                "25th_percentile",
                "median",
                "75th_percentile",
                "90th_percentile",
                "95th_percentile",
                "99th_percentile",
                "maximum",
                "mean",
                "standard_deviation",
            ],
            "value": [
                sales.min(),
                quantiles.loc[0.25],
                quantiles.loc[0.50],
                quantiles.loc[0.75],
                quantiles.loc[0.90],
                quantiles.loc[0.95],
                quantiles.loc[0.99],
                sales.max(),
                sales.mean(),
                sales.std(),
            ],
        }
    )

    # IQR is used to flag candidates for investigation,
    # not to automatically delete records.
    q1 = quantiles.loc[0.25]
    q3 = quantiles.loc[0.75]
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)

    potential_outliers = (
        df.loc[df["sales"] > upper_bound]
        .sort_values("sales", ascending=False)
        .copy()
    )

    print("\n=== SALES DISTRIBUTION ===")
    print(distribution.to_string(index=False))

    print("\n=== POTENTIAL HIGH-SALES OUTLIERS ===")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Upper IQR threshold: {upper_bound:.2f}")
    print(
        "Records above threshold: "
        f"{len(potential_outliers):,}"
    )

    print("\n=== TOP 10 SALES RECORDS ===")
    print(
        df.nlargest(10, "sales")[
            ["date", "store", "item", "sales"]
        ].to_string(index=False)
    )

    return distribution, potential_outliers


def prepare_dataset() -> None:
    """Validate, standardize, and save a separate processed dataset."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()

    # Re-run all quality checks before preparing the data.
    quality_report = inspect_data(df)

    errors = quality_report.loc[
        quality_report["severity"] == "ERROR"
    ]

    reviews = quality_report.loc[
        quality_report["severity"] == "REVIEW"
    ]

    if not errors.empty or not reviews.empty:
        print("\n=== PREPARATION BLOCKED ===")

        if not errors.empty:
            print("\nChecks requiring correction:")
            print(errors.to_string(index=False))

        if not reviews.empty:
            print("\nChecks requiring review:")
            print(reviews.to_string(index=False))

        raise ValueError(
            "Resolve ERROR and REVIEW checks before "
            "preparing the modeling dataset."
        )

    # Verify complete daily coverage for each store-product pair.
    pair_summary = inspect_time_series_completeness(df)

    if not pair_summary["date_range_complete"].all():
        raise ValueError(
            "Incomplete store-product date coverage. "
            "Investigate before preparing the dataset."
        )

    # Investigate the sales distribution before changing values.
    distribution, potential_outliers = review_sales_distribution(df)

    distribution.to_csv(
        REPORTS_DIR / "sales_distribution.csv",
        index=False,
    )

    potential_outliers.to_csv(
        REPORTS_DIR / "potential_sales_outliers.csv",
        index=False,
    )

    # Standardize data types.
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["store"] = df["store"].astype("int64")
    df["item"] = df["item"].astype("int64")
    df["sales"] = df["sales"].astype("int64")

    # Sort chronologically and consistently.
    df = df.sort_values(
        ["date", "store", "item"]
    ).reset_index(drop=True)

    # Final assertions protect the processed output.
    if df.duplicated(["date", "store", "item"]).any():
        raise ValueError(
            "Duplicate date-store-item keys detected."
        )

    if df[["date", "store", "item", "sales"]].isna().any().any():
        raise ValueError(
            "Missing values detected in required columns."
        )

    if (df["sales"] < 0).any():
        raise ValueError(
            "Negative sales detected in the processed dataset."
        )

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\n=== DATA PREPARATION COMPLETE ===")
    print(f"Processed rows: {len(df):,}")
    print(f"Processed columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(
        "Duplicate keys: "
        f"{int(df.duplicated(['date', 'store', 'item']).sum())}"
    )
    print(f"Output: {PROCESSED_DATA_PATH}")
    print("\nThe original raw dataset was not modified.")


if __name__ == "__main__":
    prepare_dataset()
