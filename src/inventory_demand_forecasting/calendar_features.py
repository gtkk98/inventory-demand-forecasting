from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sales_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_calendar_features.csv"
)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features without modifying the input."""
    
    if "date" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'date' column.")
    
    if df["date"].isna().any():
        raise ValueError("The date column contains missing values.")
    
    result = df.copy()
    
    # Convert date values to pandas datetime.
    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    )
    
    # Monday = 0, Tuesday = 1, ..., Sunday = 6.
    result["day_of_week"] = (
        result["date"].dt.dayofweek.astype("int8")
    )
    
    # Human-readable weekday name.
    result["day_name"] = result["date"].dt.day_name()
    
    # Day number within the month: 1 to 31.
    result["day_of_month"] = (
        result["date"].dt.day.astype("int8")
    )
    
    # Month number: January = 1, December = 12.
    result["month"] = result["date"].dt.month.astype("int8")
    
    # Calendar year.
    result["year"] = result["date"].dt.year.astype("int16")
    
    # Quarter: 1 to 4.
    result["quarter"] = (
        result["date"].dt.quarter.astype("int8")
    )
    
    # ISO week number: 1 to 53.
    result["week_of_year"] = (
        result["date"].dt.isocalendar().week.astype("int8")
    )
    
    # Monday-Friday = 0, Saturday-Sunday = 1.
    result["is_weekend"] = (
        result["day_of_week"].isin([5, 6]).astype("int8")
    )
    
    return result

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_PATH}"
        )
    
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    
    featured_df = add_calendar_features(df)
    
    # Save a separate artifact; do not overwrite sales_clean.csv.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(OUTPUT_PATH, index=False)
    
    calendar_columns = [
        "day_of_week",
        "day_name",
        "day_of_month",
        "month",
        "year",
        "quarter",
        "week_of_year",
        "is_weekend",
    ]
    
    print("\n=== CALENDAR FEATURE ENGINEERING ===")
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(featured_df):,}")
    print(f"New features: {calendar_columns}")

    print("\n=== FEATURE PREVIEW ===")
    print(
        featured_df[
            ["date", "store", "item", "sales", *calendar_columns]
        ].head(10).to_string(index=False)
    )

    print("\n=== FEATURE DATA TYPES ===")
    print(featured_df[calendar_columns].dtypes.to_string())

    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
