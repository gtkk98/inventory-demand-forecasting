from pathlib import Path

import numpy as np
import pandas as pd

from inventory_demand_forecasting.data import RAW_DATA_PATH

REQUIRED_COLUMNS = { "date", "store", "item", "sales" }
KEY_COLUMNS = ["date", "store", "item"]

def validate_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run data-quality checks on a sales DataFrame.

    Returns:
        A DataFrame containing one row per validation check.
        The function reports problems without deleting or modifying
        the original DataFrame.
    """
    results = []
    
    def add_result(
        check: str,
        issue_count: int,
        status: str,
        details: str,
    ) -> None:
        results.append(
            {
                "check": check,
                "issue_count": int(issue_count),
                "status": status,
                "details": details,
            }
        )
        
    # Check 1: Required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
        
    add_result(
        "Required columns",
        len(missing_columns),
        "PASS" if not missing_columns else "FAILS",
        (
            "All required columns exist."
            if not missing_columns
            else f"Missing columns: {sorted(missing_columns)}"
        ),
    )
    
    # Stop checks that depend on missing columns.
    if missing_columns:
        return pd.DataFrame(results)
    
    # Check 2: Missing values
    missing_values = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())
    
    add_result(
        "Missing required values",
        missing_values,
        "PASS" if missing_values == 0 else "FAIL",
        f"Total missing cells in required columns: {missing_values}",
    )
    
    # Check 3: Date parsing
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    
    add_result(
        "Invalid dates",
        invalid_dates,
        "PASS" if invalid_dates == 0 else "FAIL",
        f"Rows with unparseable or missing dates: {invalid_dates}",
    )
    
    # Check 4: Store and product identifiers
    for column in ["store", "item"]:
        numeric_ids = pd.to_numeric(df[column], errors="coerce")
        
        invalid_ids = (
            numeric_ids.isna()
            | ~np.isfinite(numeric_ids.fillna(0))
            | (numeric_ids <= 0)
            | (numeric_ids % 1 != 0)
        )
        
        count = int(invalid_ids.sum())
        
        add_result(
            f"Valid {column} identifiers",
            count,
            "PASS" if count == 0 else "FAIL",
            f"Invalid {column} identifiers: {count}",
        )
        
    # Check 5: Sales values must be numeric and finite
    numeric_sales = pd.to_numeric(df["sales"], errors="coerce")
    
    non_numeric = int(numeric_sales.isna().sum())
    non_finite = int(
        (~np.isfinite(numeric_sales.fillna(0))).sum()
    )
    
    add_result(
        "Numeric sales values",
        non_numeric,
        "PASS" if non_numeric == 0 else "FAIL",
        f"Missing or non-numeric sales values: {non_numeric}",
    )
    
    add_result(
        "Finite sales values",
        non_finite,
        "PASS" if non_finite == 0 else "FAIL",
        f"Infinite or negative-infinite sales values: {non_finite}",
    )
    
    # Check 6: Negative sales
    negative_sales = int((numeric_sales < 0).fillna(False).sum())
    
    add_result(
        "Non-negative sales",
        negative_sales,
        "PASS" if negative_sales == 0 else "FAIL",
        f"Rows with negative sales: {negative_sales}",
    )
    
    # Check 7: Fractional sales
    valid_numeric_sales = numeric_sales.dropna()
    finite_sales = valid_numeric_sales[
        np.isfinite(valid_numeric_sales)
    ] 
    fractional_sales = int(
        (finite_sales % 1 != 0).sum()
    )
    
    add_result(
        "Whole-number sales",
        fractional_sales,
        "PASS" if fractional_sales == 0 else "WARNING",
        (
            f"Rows with fractional sales: {fractional_sales}. "
            "Review whether sales represent individual units"
        ),
    )
    
    # Check 8: Duplicate store-product-date keys
    duplicate_keys = int(
        df.duplicated(subset=KEY_COLUMNS, keep=False).sum()
    )
    
    add_result(
        "Unique store-product-date keys",
        duplicate_keys,
        "PASS" if duplicate_keys == 0 else "FAIL",
        (
            "No duplicate store-product-date keys."
            if duplicate_keys == 0
            else (
                f"{duplicate_keys} rows belong to duplicate "
                "store-product-date groups."
            )
        ),
    )

    # Check 9: Date range
    valid_dates = parsed_dates.dropna()

    if valid_dates.empty:
        add_result(
            "Date range",
            0,
            "WARNING",
            "No valid dates available to determine the date range.",
        )
    else:
        add_result(
            "Date range",
            0,
            "PASS",
            (
                f"From {valid_dates.min().date()} "
                f"to {valid_dates.max().date()}."
            ),
        )

    return pd.DataFrame(results)


def check_time_series_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check missing daily observations for each store-item pair.

    Assumes each observed store-item pair should have one record
    per calendar day across the dataset's global date range.
    """
    required = REQUIRED_COLUMNS

    if not required.issubset(df.columns):
        raise ValueError(
            f"Required columns are: {sorted(required)}"
        )

    if df.empty:
        raise ValueError("Cannot check gaps in an empty dataset.")

    work = df[["date", "store", "item"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    work = work.dropna(subset=["date", "store", "item"])
    work["date"] = work["date"].dt.normalize()

    if work.empty:
        raise ValueError("No valid dates or identifiers to check.")

    start_date = work["date"].min()
    end_date = work["date"].max()

    expected_days = len(
        pd.date_range(start=start_date, end=end_date, freq="D")
    )

    # Duplicate keys should be resolved before interpreting gaps.
    observed = work.drop_duplicates(
        subset=["date", "store", "item"]
    )

    coverage = (
        observed.groupby(["store", "item"])["date"]
        .nunique()
        .rename("observed_days")
        .reset_index()
    )

    coverage["expected_days"] = expected_days
    coverage["missing_days"] = (
        coverage["expected_days"] - coverage["observed_days"]
    )

    coverage["has_gaps"] = coverage["missing_days"] > 0

    return coverage.sort_values(
        "missing_days",
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    """Run the validation checks and display the report."""
    # Read raw values so the validator can report malformed dates
    # instead of failing during strict date conversion.
    df = pd.read_csv(RAW_DATA_PATH)

    report = validate_sales_data(df)

    print("\n=== DATA QUALITY REPORT ===")
    print(report.to_string(index=False))

    output_path = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "data_quality_report.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print(f"\nReport saved to: {output_path}")

    # Check the completeness of daily store-item observations.
    gaps = check_time_series_gaps(df)

    total_missing = int(gaps["missing_days"].sum())
    affected_series = int(gaps["has_gaps"].sum())

    print("\n=== TIME-SERIES COMPLETENESS ===")
    print(f"Store-item series: {len(gaps):,}")
    print(f"Series with missing days: {affected_series:,}")
    print(f"Missing store-item-day observations: {total_missing:,}")

    if affected_series:
        print("\nTop 10 store-item series with missing days:")
        print(gaps.head(10).to_string(index=False))
    else:
        print("All observed store-item series have complete coverage.")


if __name__ == "__main__":
    main()