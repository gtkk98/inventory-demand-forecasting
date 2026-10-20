from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_rolling_features.csv"
)

TRAIN_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_train.csv"
)

TEST_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_test.csv"
)

TRAIN_END_DATE = pd.Timestamp("2016-12-31")
TEST_START_DATE = pd.Timestamp("2017-01-01")
TEST_END_DATE = pd.Timestamp("2017-12-31")

def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forecasting data chronologically without shuffling."""
    
    required_columns = {
        "date",
        "store",
        "item",
        "sales",
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_mean_7",
        "sales_rolling_mean_30",
        "sales_rolling_std_7",
    }
    
    missing_columns = required_columns.difference(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    
    if result["date"].isna().any():
        raise ValueError("The dataset contains missing dates.")
    
    if result.duplicated(["date", "store", "item"]).any():
        raise ValueError(
            "Duplicate date-store-item records must be resolved first."
        )
    
    train_df = result.loc[
        result["date"] <= TRAIN_END_DATE
    ].copy()
    
    test_df = result.loc[
        result["date"].between(TEST_START_DATE, TEST_END_DATE)
    ].copy()
    
    if train_df.empty or test_df.empty:
        raise ValueError(
            "The training or testing dataset is empty. "
            "Check the available date range."
        )

    # Confirm that the split does not overlap in time.
    if train_df["date"].max() >= test_df["date"].min():
        raise ValueError(
            "Temporal leakage detected: training dates overlap "
            "with testing dates."
        )

    # Confirm the intended date ranges.
    if train_df["date"].max() != TRAIN_END_DATE:
        raise ValueError(
            f"Expected training to end on {TRAIN_END_DATE.date()}."
        )

    if test_df["date"].min() != TEST_START_DATE:
        raise ValueError(
            f"Expected testing to start on {TEST_START_DATE.date()}."
        )

    if test_df["date"].max() != TEST_END_DATE:
        raise ValueError(
            f"Expected testing to end on {TEST_END_DATE.date()}."
        )

    return (
        train_df.sort_values(
            ["date", "store", "item"]
        ).reset_index(drop=True),
        test_df.sort_values(
            ["date", "store", "item"]
        ).reset_index(drop=True),
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    train_df, test_df = split_dataset(df)

    TRAIN_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("\n=== CHRONOLOGICAL TRAIN/TEST SPLIT ===")

    print("\nTraining dataset")
    print(f"Rows: {len(train_df):,}")
    print(f"Start date: {train_df['date'].min().date()}")
    print(f"End date:   {train_df['date'].max().date()}")

    print("\nTesting dataset")
    print(f"Rows: {len(test_df):,}")
    print(f"Start date: {test_df['date'].min().date()}")
    print(f"End date:   {test_df['date'].max().date()}")

    print("\n=== SPLIT VALIDATION ===")
    print(
        "Training ends before testing starts:",
        train_df["date"].max() < test_df["date"].min(),
    )
    print(
        "Total rows preserved:",
        len(train_df) + len(test_df) == len(df),
    )

    print(f"\nTraining file saved to: {TRAIN_OUTPUT_PATH}")
    print(f"Testing file saved to:  {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
