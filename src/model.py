from __future__ import annotations
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from joblib import dump, load
from .features import auto_preprocess, split_features_label

def logistic_pipeline():
    clf = LogisticRegression(max_iter=200)
    pipe = Pipeline(steps=[
        ("prep", None),
        ("clf", clf)
    ])
    return pipe
    
def decision_tree_pipeline():
    dec_tree_model = DecisionTreeClassifier()
    pipe = Pipeline(steps=[
        ("prep", None),
        ("clf", clf)
    ])
    return dec_tree_model
    
def ensemble_decision_tree_pipeline():
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': { 'binary_logloss'},
        'num_leaves': 31, # defauly leaves(31) amount for each tree
        'learning_rate': 0.03,
        'feature_fraction': 0.7, # will select 70% features before training each tree
        'bagging_fraction': 0.3, #feature_fraction, but this will random select part of data
        'bagging_freq': 5, #  perform bagging at every 5 iteration
        'verbose': 0
    }
    
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=500)
