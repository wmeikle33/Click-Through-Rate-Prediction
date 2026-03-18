import argparse
from pathlib import Path

from .data import load_csv
from .model import train_eval_save

DEFAULT_DATA = "data/raw/train.csv"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=DEFAULT_DATA,
        help="Path to training CSV (default: data/raw/train.csv)",
    )
    ap.add_argument("--label", default="click", help="Target column")
    ap.add_argument(
        "--model-out",
        default="models/model.joblib",
        help="Saved model path",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_path = Path(args.model_out).expanduser().resolve()

    df = load_csv(csv_path)

    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in {csv_path}")

    metrics = train_eval_save(
        df=df,
        label=args.label,
        model_path=str(model_path),
        random_state=args.random_state,
        test_size=args.test_size,
    )

    print(f"Saved model to: {model_path}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    if "auc" in metrics:
        print(f"auc={metrics['auc']:.6f}")


if __name__ == "__main__":
    main()
