from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_calendar_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_with_lag_features.csv"
)

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical sales features separately for each store-item pair."""
     
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
        errors="raise"
    )
    
    # Every store-item combination must have at most one row per date.
    duplicate_mask = result.duplicated(
        subset=["date", "store", "item"],
        keep=False,
    )
    
    if duplicate_mask.any():
        raise ValueError(
            "Duplicate date-store-item records found. "
            "Resolve duplicates before creating lag features."
        )
    
    # Sort chronologically within each store-item series.
    result = result.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)
    
    # Retrieve sales from the previous available row in each series.
    grouped_sales = result.groupby(
        ["store", "item"],
        sort=False,
    )["sales"]
    
    result["sales_lag_1"] = grouped_sales.shift(1)
    result["sales_lag_7"] = grouped_sales.shift(7)
    
    return result

# Load the sales dataset, validate that the file exists, generate lag features, 
# create the output directory if necessary, and save the processed dataset to a new CSV file.
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )
    
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )
    
    featured_df = add_lag_features(df)
    
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    featured_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )
    
    print("\n=== LAG FEATURE ENGINEERING ===")
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(featured_df):,}")

    print("\n=== LAG FEATURE PREVIEW ===")
    print(
        featured_df[
            [
                "date",
                "store",
                "item",
                "sales",
                "sales_lag_1",
                "sales_lag_7",
            ]
        ].head(12).to_string(index=False)
    )

    print("\n=== MISSING LAG VALUES ===")
    print(
        featured_df[
            ["sales_lag_1", "sales_lag_7"]
        ].isna().sum().to_string()
    )

    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
