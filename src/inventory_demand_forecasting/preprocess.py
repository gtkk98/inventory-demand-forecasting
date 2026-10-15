from pathlib import Path
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(df["date"])

    return df


def basic_data_quality_check(
    df: pd.DataFrame
) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
    }