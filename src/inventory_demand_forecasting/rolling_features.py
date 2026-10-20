from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_lag_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_rolling_features.csv"
)

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical rolling statistics for each store-item series."""
    
    required_columns = {"date", "store", "item", "sales"}
    missing_columns = required_columns.difference(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Input DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )
    
    if df["date"].isna().any():
        raise ValueError("The date column contains missing values.")
    
    if df[["store", "item", "sales"]].isna().any().any():
        raise ValueError(
            "Store, item, and sales columns must not contain missing values."
        )
    
    result = df.copy()
    
    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    )
    
    duplicate_mask = result.duplicated(
        subset=["date", "store", "item"],
        keep=False,
    )
    
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate date-store-item records found. "
            "Resolve duplicates before creating rolling features."
        )
        
    # Sort each store-item series chronologically.
    result = result.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)
    
    # Grouping prevents sales from different store-item combinations
    # from being mixed together.
    grouped_sales = result.groupby(
        ["store", "item"],
        sort=False
    )["sales"]
    
    # Shift first to exclude the current day's actual sales.
    pervious_sales = grouped_sales.shift(1)
    
    # Seven previous observations: the previous seven days.
    result["sales_rolling_mean_7"] = (
        pervious_sales
        .groupby(
            [
                result["store"],
                result["item"]
            ],
            sort=False,
        )
        .rolling(window=7, min_periods=7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    
    # Thirty previous observations: the previous thirty days.
    result["sales_rolling_mean_30"] = (
        pervious_sales
        .groupby(
            [
                result["store"],
                result["item"],
            ],
            sort=False,
        )
        .rolling(window=30, min_periods=30)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    
    # Sample standard deviation across the previous seven days.
    result["sales_rolling_std_7"] = (
        pervious_sales
        .groupby(
            [
                result["store"],
                result["item"],
            ],
            sort=False,
        )
        .rolling(window=7, min_periods=7)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )
    
    return result

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )
    
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )
    
    featured_df = add_rolling_features(df)
    
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    featured_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )
    
    rolling_columns = [
        "sales_rolling_mean_7",
        "sales_rolling_mean_30",
        "sales_rolling_std_7",
    ]
    
    print("\n=== ROLLING FEATURE ENGINEERING ===")
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(featured_df):,}")
    print(f"New features: {rolling_columns}")

    print("\n=== FEATURE PREVIEW ===")
    print(
        featured_df[
            [
                "date",
                "store",
                "item",
                "sales",
                *rolling_columns,
            ]
        ].head(40).tail(12).to_string(index=False)
    )

    print("\n=== MISSING ROLLING VALUES ===")
    print(
        featured_df[rolling_columns]
        .isna()
        .sum()
        .to_string()
    )

    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()