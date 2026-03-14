import pandas as pd
from ctr_prediction.model import time_based_split


def test_time_based_split_uses_latest_day_for_validation():
    df = pd.DataFrame(
        {
            "hour": [14102100, 14102101, 14102200, 14102300],
            "click": [0, 1, 0, 1],
            "feature_a": ["a", "b", "c", "d"],
        }
    )

    X_train, X_valid, y_train, y_valid = time_based_split(df, label="click", hour_col="hour")

    assert len(X_train) == 3
    assert len(X_valid) == 1
    assert X_valid["hour"].iloc[0] == 14102300
