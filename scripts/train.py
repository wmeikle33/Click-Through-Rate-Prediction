#!/usr/bin/env python
"""Train a baseline CTR model from a CSV."""
import argparse
from pathlib import Path
from src.data import load_csv
from src.model import train_eval_save
from src.config import Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to training CSV")
    ap.add_argument("--label", required=True, help="Label column name")
    args = ap.parse_args()

    cfg = Config()
    df = load_csv(args.csv)
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    metrics = train_eval_save(df, args.label, str(cfg.model_path), str(cfg.pipeline_path), cfg.random_state)
    print("Saved model to:", cfg.model_path)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
