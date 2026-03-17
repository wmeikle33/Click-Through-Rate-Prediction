import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from .data import load_csv
from .features import split_features_label
from .model import train_eval_save
DEFAULT_DATA = "data/raw/train.csv"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_DATA), help="Path to training CSV (default: data/raw/train.csv)")
    ap.add_argument("--label", default="click", help="Target column")
    ap.add_argument("--model-out", default="models/model.joblib", help="Saved model path")
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model_out).expanduser().resolve()

    df = load_csv(csv_path)
    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in {args.csv}")

    X, y = split_features_label(df, args.label)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    metrics = train_eval_save(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        model_path=args.model_out,
    )

    print(f"Saved model to: {Path(args.model_out)}")
    print(f"log_loss={metrics['log_loss']:.6f}")
    print(f"roc_auc={metrics['roc_auc']:.6f}")


if __name__ == "__main__":
    main()
