#!/usr/bin/env python
"""Load saved model and predict for a CSV of new samples."""
import argparse
import pandas as pd
from joblib import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/model.joblib", help="Path to saved model (joblib)")
    ap.add_argument("--input", required=True, help="CSV file with the same feature columns used in training")
    ap.add_argument("--output", default="predictions.csv", help="Where to save predictions (CSV)")
    args = ap.parse_args()

    pipe = load(args.model)
    df = pd.read_csv(args.input)
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(df)[:,1]
        out = df.copy()
        out["prediction"] = pipe.predict(df)
        out["probability"] = prob
    else:
        out = df.copy()
        out["prediction"] = pipe.predict(df)
    out.to_csv(args.output, index=False)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
