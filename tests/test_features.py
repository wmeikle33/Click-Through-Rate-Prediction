import pandas as pd

from ctr_prediction.features import split_features_label, auto_preprocess


def test_split_features_label_splits_target_correctly():
    df = pd.DataFrame(
        {
            "hour": [1, 2, 3],
            "site_id": ["a", "b", "a"],
            "click": [0, 1, 0],
        }
    )

    X, y = split_features_label(df, label="click")

    assert "click" not in X.columns
    assert list(X.columns) == ["hour", "site_id"]
    assert y.tolist() == [0, 1, 0]


def test_auto_preprocess_runs_on_small_mixed_dataframe():
    X = pd.DataFrame(
        {
            "hour": [1, 2, 3],
            "site_id": ["a", "b", "a"],
        }
    )

    preprocessor = auto_preprocess(X)
    Xt = preprocessor.fit_transform(X)

    assert Xt.shape[0] == 3
