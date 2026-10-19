
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
REPORTS_PATH = PROJECT_ROOT / "reports"

REQUIRED_COLUMNS = ["date", "store", "item", "sales"]


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV without silently discarding invalid values."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create a report of potential data-quality problems."""
    report = []

    def add_check(
        check_name: str,
        affected_records: int,
        severity: str,
        details: str,
    ) -> None:
        report.append(
            {
                "check": check_name,
                "affected_records": int(affected_records),
                "severity": severity,
                "details": details,
            }
        )

    # 1. Missing values in required columns.
    for column in REQUIRED_COLUMNS:
        count = df[column].isna().sum()

        add_check(
            f"Missing values: {column}",
            count,
            "ERROR" if count else "PASS",
            f"Number of missing values in '{column}'.",
        )

    # 2. Invalid dates.
    parsed_dates = pd.to_datetime(
        df["date"], errors="coerce"
    )
    invalid_dates = parsed_dates.isna().sum()

    add_check(
        "Invalid dates",
        invalid_dates,
        "ERROR" if invalid_dates else "PASS",
        "Non-missing date values that cannot be parsed "
        "are also counted here.",
    )

    # 3. Store and product identifiers.
    for column in ["store", "item"]:
        numeric_ids = pd.to_numeric(
            df[column], errors="coerce"
        )

        invalid_ids = (
            numeric_ids.isna()
            | ~np.isfinite(numeric_ids)
            | (numeric_ids <= 0)
            | (numeric_ids % 1 != 0)
        )

        add_check(
            f"Invalid identifiers: {column}",
            invalid_ids.sum(),
            "ERROR" if invalid_ids.any() else "PASS",
            "IDs should be positive, finite whole numbers.",
        )

    # 4. Sales values that cannot be interpreted as numbers.
    numeric_sales = pd.to_numeric(
        df["sales"], errors="coerce"
    )

    invalid_sales = (
        numeric_sales.isna()
        | ~np.isfinite(numeric_sales)
    )

    add_check(
        "Invalid or non-finite sales",
        invalid_sales.sum(),
        "ERROR" if invalid_sales.any() else "PASS",
        "Includes missing, non-numeric, infinite, "
        "and negative-infinity sales values.",
    )

    # 5. Negative sales.
    negative_sales = (
        numeric_sales.notna() & (numeric_sales < 0)
    )

    add_check(
        "Negative sales",
        negative_sales.sum(),
        "REVIEW" if negative_sales.any() else "PASS",
        "Investigate before deciding whether to keep "
        "or remove these records.",
    )

    # 6. Fractional sales.
    fractional_sales = (
        numeric_sales.notna()
        & np.isfinite(numeric_sales)
        & (numeric_sales % 1 != 0)
    )

    add_check(
        "Fractional sales",
        fractional_sales.sum(),
        "REVIEW" if fractional_sales.any() else "PASS",
        "Sales are expected to represent whole units "
        "in this dataset.",
    )

    # 7. Duplicate date-store-item combinations.
    duplicate_keys = df.duplicated(
        subset=["date", "store", "item"],
        keep=False,
    )

    add_check(
        "Duplicate date-store-item records",
        duplicate_keys.sum(),
        "ERROR" if duplicate_keys.any() else "PASS",
        "Counts every row belonging to a duplicated "
        "key group, not just the extra rows.",
    )

    # 8. Basic dataset size.
    add_check(
        "Total records",
        len(df),
        "INFO",
        "Total number of rows in the raw dataset.",
    )

    return pd.DataFrame(report)


def main() -> None:
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    report = inspect_data(df)

    output_path = REPORTS_PATH / "cleaning_inspection.csv"
    report.to_csv(output_path, index=False)
    
    pair_summary = inspect_time_series_completeness(df)

    completeness_path = (
        REPORTS_PATH / "time_series_completeness.csv"
    )
    pair_summary.to_csv(completeness_path, index=False)

    print(
        f"\nCompleteness report saved to: "
        f"{completeness_path}"
    )

    print("\n=== CLEANING INSPECTION REPORT ===")
    print(report.to_string(index=False))

    print(f"\nTotal rows inspected: {len(df):,}")
    print(f"Report saved to: {output_path}")
    print("\nNo raw data has been modified.")
    
    
def inspect_time_series_completeness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether every store-product pair has daily records."""
    work = df.copy()

    work["date"] = pd.to_datetime(
        work["date"], errors="coerce"
    )

    # Invalid keys must be investigated separately.
    work = work.dropna(subset=["date", "store", "item"])

    if work.empty:
        raise ValueError(
            "No valid date-store-product records to inspect."
        )

    start_date = work["date"].min().normalize()
    end_date = work["date"].max().normalize()

    expected_days = len(
        pd.date_range(start=start_date, end=end_date, freq="D")
    )

    duplicate_mask = work.duplicated(
        subset=["date", "store", "item"],
        keep=False,
    )

    duplicate_rows = int(duplicate_mask.sum())

    pair_summary = (
        work.groupby(["store", "item"])["date"]
        .agg(
            observed_dates="nunique",
            first_date="min",
            last_date="max",
        )
        .reset_index()
    )

    pair_summary["expected_dates"] = expected_days
    pair_summary["missing_dates"] = (
        pair_summary["expected_dates"]
        - pair_summary["observed_dates"]
    ).clip(lower=0)

    pair_summary["date_range_complete"] = (
        (pair_summary["first_date"] == start_date)
        & (pair_summary["last_date"] == end_date)
        & (pair_summary["observed_dates"] == expected_days)
    )

    print("\n=== TIME-SERIES INTEGRITY ===")
    print(f"First date: {start_date.date()}")
    print(f"Last date: {end_date.date()}")
    print(f"Expected calendar days: {expected_days:,}")
    print(
        f"Store-product combinations: "
        f"{len(pair_summary):,}"
    )
    print(
        f"Expected records in complete grid: "
        f"{expected_days * len(pair_summary):,}"
    )
    print(
        f"Observed records: {len(work):,}"
    )
    print(
        f"Rows belonging to duplicate key groups: "
        f"{duplicate_rows:,}"
    )
    print(
        f"Store-product pairs with missing dates: "
        f"{(~pair_summary['date_range_complete']).sum():,}"
    )

    return pair_summary



if __name__ == "__main__":
    main()
