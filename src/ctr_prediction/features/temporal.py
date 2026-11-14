import pandas as pd


def extract_time_features(df):
    df = df.copy()

    timestamp = pd.to_datetime(
        df["hour"].astype(str),
        format="%y%m%d%H"
    )

    df["hour_of_day"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    return df
