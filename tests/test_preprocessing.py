import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
assert len(X_train) == 3
assert len(X_valid) == 1
assert X_valid["hour"].iloc[0] == 14102300

