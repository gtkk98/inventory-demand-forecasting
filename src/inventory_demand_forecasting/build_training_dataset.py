from pathlib import Path

import pandas as pd

from preprocess import load_data

from feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_expanding_features,
)

def main():

    df = load_data(
        "data/raw/train.csv"
    )

    df = create_time_features(df)

    df = create_lag_features(
        df,
        [7, 14, 30]
    )

    df = create_rolling_features(df)

    df = create_expanding_features(df)
    
    df = df.dropna()
    
    df.to_parquet(
        "data/processed/train_features.parquet",
        index=False,
    )
    
    print(df.shape)
    
    print(df.head())

if __name__ == "__main__":
    main()