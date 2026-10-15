import pandas as pd

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    df["day"] = df["date"].dt.day

    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = (
        df["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    return df

def create_lag_features(
    df: pd.DataFrame,
    lags: list[int]
) -> pd.DataFrame:

    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(
                ["store", "item"]
            )["sales"]
            .shift(lag)
        )

    return df

def create_rolling_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    for window in [7, 14, 30]:

        df[f"rolling_mean_{window}"] = (
            df.groupby(
                ["store", "item"]
            )["sales"]
            .transform(
                lambda x:
                x.shift(1)
                 .rolling(window)
                 .mean()
            )
        )

    return df

def create_expanding_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["expanding_mean"] = (
        df.groupby(
            ["store", "item"]
        )["sales"]
        .transform(
            lambda x:
            x.shift(1)
             .expanding()
             .mean()
        )
    )

    return df