from pathlib import Path
 
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"
)
REPORTS_PATH = PROJECT_ROOT / "reports"

REQUIRED_COLUMNS = ["date", "store", "item", "sales"]

def validate_processed_data() -> pd.DataFrame:
    """Validate the saved dataset before feature engineering."""
    if not PROCESSES_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSES_PATH}")
     
    df = pd.read_csv(
        PROCESSES_PATH,
        parse_dates=["date"],
    )
    
    checks = []
    
    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )
        
        
    # 1. Schema
    add_check(
        "Required columns",
        list(df.columns) == REQUIRED_COLUMNS,
        f"Actual columns: {list(df.columns)}",
    )

    # 2. Expected dataset size
    add_check(
        "Expected row count",
        len(df) == 913_000,
        f"Observed {len(df):,} rows; expected 913,000.",
    )

    # 3. Missing values
    missing_count = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    add_check(
        "No missing required values",
        missing_count == 0,
        f"Missing required values: {missing_count:,}.",
    )

    # 4. Identifier and sales types
    integer_types_valid = all(
        pd.api.types.is_integer_dtype(df[column])
        for column in ["store", "item", "sales"]
    )
    add_check(
        "Integer column types",
        integer_types_valid,
        "store, item, and sales must use integer dtypes.",
    )

    # 5. Valid domain values
    valid_values = (
        (df["store"] > 0).all()
        and (df["item"] > 0).all()
        and (df["sales"] >= 0).all()
        and df["date"].notna().all()
    )
    add_check(
        "Valid dates, IDs, and sales",
        bool(valid_values),
        "Dates must be valid, IDs positive, and sales non-negative.",
    )

    # 6. Unique keys
    duplicate_count = int(
        df.duplicated(["date", "store", "item"]).sum()
    )
    add_check(
        "Unique date-store-item keys",
        duplicate_count == 0,
        f"Duplicate keys: {duplicate_count:,}.",
    )

    # 7. Chronological sorting
    correctly_sorted = (
        df.equals(
            df.sort_values(
                ["date", "store", "item"]
            ).reset_index(drop=True)
        )
    )
    add_check(
        "Chronological ordering",
        bool(correctly_sorted),
        "Rows should be ordered by date, store, and item.",
    )

    # 8. Complete daily coverage
    if not df.empty and df["date"].notna().all():
        expected_days = len(
            pd.date_range(
                df["date"].min(),
                df["date"].max(),
                freq="D",
            )
        )

        pair_summary = (
            df.groupby(["store", "item"])["date"]
            .nunique()
        )

        complete = (
            len(pair_summary) == 500
            and (pair_summary == expected_days).all()
        )

        add_check(
            "Complete daily store-product coverage",
            bool(complete),
            (
                f"{len(pair_summary)} combinations; "
                f"{expected_days} expected dates per combination."
            ),
        )
    else:
        add_check(
            "Complete daily store-product coverage",
            False,
            "Cannot verify date coverage for an empty dataset "
            "or invalid dates.",
        )

    report = pd.DataFrame(checks)

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_PATH / "processed_data_validation.csv"
    report.to_csv(report_path, index=False)

    print("\n=== PROCESSED DATA VALIDATION ===")
    print(report.to_string(index=False))
    print(f"\nValidation report saved to: {report_path}")

    failed = report.loc[report["status"] == "FAIL"]

    if not failed.empty:
        raise ValueError(
            "Processed dataset validation failed. "
            "Review the failed checks before proceeding."
        )

    print("\nAll validation checks passed.")

    return report


if __name__ == "__main__":
    validate_processed_data()
  