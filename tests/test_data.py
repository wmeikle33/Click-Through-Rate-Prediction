import pandas as pd

from ctr_prediction.data import load_csv, save_csv


def test_save_and_load_csv_roundtrip(tmp_path):
    df = pd.DataFrame(
        {
            "hour": [1, 2, 3],
            "site_id": ["a", "b", "a"],
            "click": [0, 1, 0],
        }
    )

    out_path = tmp_path / "sample.csv"
    save_csv(df, out_path)

    loaded = load_csv(out_path)

    pd.testing.assert_frame_equal(loaded, df)
