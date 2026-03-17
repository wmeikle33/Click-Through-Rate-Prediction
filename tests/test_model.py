import pandas as pd

from ctr_prediction.model import build_pipeline, train_eval_save, load_model


def make_small_training_df():
    return pd.DataFrame(
        {
            "hour": [1, 2, 3, 4, 5, 6],
            "site_id": ["a", "b", "a", "b", "c", "c"],
            "click": [0, 1, 0, 1, 0, 1],
        }
    )


def test_build_pipeline_can_fit_and_predict():
    df = make_small_training_df()
    X = df.drop(columns=["click"])
    y = df["click"]

    pipe = build_pipeline(X)
    pipe.fit(X, y)
    preds = pipe.predict_proba(X)[:, 1]

    assert preds.shape == (len(X),)


def test_train_eval_save_writes_model_and_returns_metrics(tmp_path):
    df = make_small_training_df()
    model_path = tmp_path / "model.joblib"

    metrics = train_eval_save(
        df=df,
        label="click",
        model_path=str(model_path),
        random_state=42,
        test_size=0.33,
    )

    assert model_path.exists()
    assert "accuracy" in metrics


def test_load_model_loads_saved_pipeline(tmp_path):
    df = make_small_training_df()
    model_path = tmp_path / "model.joblib"

    train_eval_save(
        df=df,
        label="click",
        model_path=str(model_path),
        random_state=42,
        test_size=0.33,
    )

    model = load_model(str(model_path))
    X = df.drop(columns=["click"])
    preds = model.predict_proba(X)[:, 1]

