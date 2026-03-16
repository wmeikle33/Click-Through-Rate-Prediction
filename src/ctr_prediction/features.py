from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def split_features_label(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    y = df[label]
    X = df.drop(columns=[label])
    return X, y

def auto_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=False), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    return ColumnTransformer(transformers)
