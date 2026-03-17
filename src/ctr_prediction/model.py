from __future__ import annotations
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .features import auto_preprocess, split_features_label


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = auto_preprocess(X)
    classifier = LogisticRegression(max_iter=200)

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", classifier),
        ]
    )

def train_eval_save(
    df: pd.DataFrame,
    label: str,
    model_path: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, float]:
    X, y = split_features_label(df, label)

    pipe = build_pipeline(X)

    stratify = y if y.nunique() <= 20 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_val, y_pred))
    }

    if hasattr(pipe, "predict_proba") and y.nunique() == 2:
        y_prob = pipe.predict_proba(X_val)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_val, y_prob))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_path)

    return metrics


def load_model(path: str):
    return load(path)
