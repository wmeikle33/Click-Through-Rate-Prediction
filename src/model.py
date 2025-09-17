from __future__ import annotations
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump, load
from .features import auto_preprocess, split_features_label

def build_pipeline():
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline(steps=[
        ("prep", None),
        ("clf", clf)
    ])
    return pipe

def train_eval_save(df, label: str, model_path: str, pipeline_path: str, random_state: int = 42):
    X, y = split_features_label(df, label)
    prep = auto_preprocess(X)
    pipe = build_pipeline()
    pipe.steps[0] = ("prep", prep)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y if len(getattr(y, 'unique', lambda: [])())<=20 else None)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {"accuracy": float(accuracy_score(y_test, y_pred))}
    if hasattr(pipe, "predict_proba") and len(getattr(y_test, 'unique', lambda: set())()) == 2:
        try:
            y_prob = pipe.predict_proba(X_test)[:,1]
            metrics["auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            pass

    dump(pipe, model_path)
    dump(prep, pipeline_path)
    return metrics

def load_model(path: str):
    return load(path)
