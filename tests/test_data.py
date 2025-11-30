from ctr_prediction.data import train_valid_split
import pandas as pd

def test_train_valid_split_respects_fraction():
    df = pd.DataFrame({"x": range(100)})
    train_df, valid_df = train_valid_split(df, valid_frac=0.2)
    assert len(train_df) == 80
    assert len(valid_df) == 20
