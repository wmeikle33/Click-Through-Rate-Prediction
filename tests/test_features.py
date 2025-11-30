from ctr_prediction.features import make_pipeline
import pandas as pd

def test_pipeline_runs_on_small_df():
    df = pd.DataFrame({
        "hour": [1, 2, 3],
        "site_id": ["a", "b", "a"],
        "click": [0, 1, 0],
    })
    X = df[["hour", "site_id"]]
    y = df["click"]

    pipe = make_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict_proba(X)[:, 1]
    assert preds.shape == (3,)
