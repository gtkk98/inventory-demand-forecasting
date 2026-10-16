from pathlib import Path

import pandas as pd

# Locate the project root independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of the original dataset
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

def load_sales_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the historical sales dataset from a CSV file.

    Args:
        file_path: Path to the source CSV file.

    Returns:
        A pandas DataFrame containing the sales records.

    Raises:
        FileNotFoundError: If the specified CSV does not exist.
        ValueError: If the CSV has no rows or required columns.
    """
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("The sales dataset contains no records.")
    
    required_columns = {"date", "store", "item", "sales"}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    # Convert date strings into Pandas datetime values.
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    
    return df

def main() -> None:
    """Load the dataset and display a basic inspection report."""
    df = load_sales_data()

    print("\n=== DATASET LOADED SUCCESSFULLY ===")
    print(f"Dataset path: {RAW_DATA_PATH}")
    print(f"Number of rows: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")

    print("\n=== FIRST FIVE RECORDS ===")
    print(df.head().to_string(index=False))

    print("\n=== COLUMN DATA TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== DATASET INFORMATION ===")
    df.info()

    print("\n=== NUMERICAL SUMMARY ===")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()