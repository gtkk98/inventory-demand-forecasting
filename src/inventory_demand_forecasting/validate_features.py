
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_rolling_features.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "feature_validation_report.csv"
)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the engineered forecasting dataset."""

    checks = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    required_columns = {
        "date",
        "store",
        "item",
        "sales",
        "day_of_week",
        "day_name",
        "day_of_month",
        "month",
        "year",
        "quarter",
        "week_of_year",
        "is_weekend",
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_mean_7",
        "sales_rolling_mean_30",
        "sales_rolling_std_7",
    }

    missing_columns = sorted(required_columns - set(df.columns))

    add_check(
        "Required feature columns",
        not missing_columns,
        f"Missing columns: {missing_columns}",
    )

    if missing_columns:
        return pd.DataFrame(checks)

    add_check(
        "Expected row count",
        len(df) == 913_000,
        f"Observed {len(df):,} rows; expected 913,000.",
    )

    dates = pd.to_datetime(df["date"], errors="coerce")

    add_check(
        "Valid dates",
        dates.notna().all(),
        f"Invalid or missing dates: {int(dates.isna().sum())}",
    )

    key_columns = ["date", "store", "item"]
    duplicate_count = int(
        df.duplicated(subset=key_columns).sum()
    )

    add_check(
        "Unique date-store-item records",
        duplicate_count == 0,
        f"Duplicate records: {duplicate_count}",
    )

    expected_lag_1_missing = int(
        df.groupby(["store", "item"], sort=False).size().shape[0]
    )
    expected_lag_7_missing = expected_lag_1_missing * 7
    expected_rolling_30_missing = expected_lag_1_missing * 30

    add_check(
        "Lag 1 missing values",
        int(df["sales_lag_1"].isna().sum()) == expected_lag_1_missing,
        f"Observed {int(df['sales_lag_1'].isna().sum()):,}; "
        f"expected {expected_lag_1_missing:,}.",
    )

    add_check(
        "Lag 7 missing values",
        int(df["sales_lag_7"].isna().sum()) == expected_lag_7_missing,
        f"Observed {int(df['sales_lag_7'].isna().sum()):,}; "
        f"expected {expected_lag_7_missing:,}.",
    )

    add_check(
        "Rolling 7-day mean missing values",
        int(df["sales_rolling_mean_7"].isna().sum())
        == expected_lag_7_missing,
        f"Observed "
        f"{int(df['sales_rolling_mean_7'].isna().sum()):,}; "
        f"expected {expected_lag_7_missing:,}.",
    )

    add_check(
        "Rolling 30-day mean missing values",
        int(df["sales_rolling_mean_30"].isna().sum())
        == expected_rolling_30_missing,
        f"Observed "
        f"{int(df['sales_rolling_mean_30'].isna().sum()):,}; "
        f"expected {expected_rolling_30_missing:,}.",
    )

    add_check(
        "Rolling 7-day standard deviation missing values",
        int(df["sales_rolling_std_7"].isna().sum())
        == expected_lag_7_missing,
        f"Observed "
        f"{int(df['sales_rolling_std_7'].isna().sum()):,}; "
        f"expected {expected_lag_7_missing:,}.",
    )

    numeric_feature_columns = [
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "quarter",
        "week_of_year",
        "is_weekend",
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_mean_7",
        "sales_rolling_mean_30",
        "sales_rolling_std_7",
    ]

    # NaN values are allowed because lag and rolling features intentionally
    # have missing values at the beginning of each series.
    numeric_values = df[numeric_feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    original_values = df[numeric_feature_columns]

    # Detect non-numeric values that were converted to NaN.
    invalid_numeric_values = numeric_values.isna() & original_values.notna()

    # Detect positive and negative infinity.
    infinite_values = numeric_values.isin([float("inf"), float("-inf")])

    finite_features = (
        not invalid_numeric_values.any().any()
        and not infinite_values.any().any()
    )

    add_check(
        "No infinite or invalid numeric features",
        finite_features,
        (
            "All non-missing numeric values are valid and finite."
            if finite_features
            else (
                f"Invalid numeric values: "
                f"{int(invalid_numeric_values.sum().sum())}; "
                f"infinite values: "
                f"{int(infinite_values.sum().sum())}."
            )
        ),
    )

    nonnegative_features = [
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_mean_7",
        "sales_rolling_mean_30",
        "sales_rolling_std_7",
    ]

    nonnegative = (
        df[nonnegative_features]
        .dropna()
        .ge(0)
        .all()
        .all()
    )

    add_check(
        "Non-negative historical sales features",
        bool(nonnegative),
        "Lag values, rolling means, and rolling standard deviations "
        "must not be negative.",
    )

    return pd.DataFrame(checks)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    report = validate_features(df)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(REPORT_PATH, index=False)

    print("\n=== FEATURE VALIDATION ===")
    print(report.to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")

    if (report["status"] == "FAIL").any():
        raise SystemExit(
            "Feature validation failed. Review the report."
        )

    print("\nAll feature validation checks passed.")


if __name__ == "__main__":
    main()
