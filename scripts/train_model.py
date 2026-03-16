import argparse
from pathlib import Path
from src.ctr_prediction.data import load_csv
from src.ctr_prediction.model import train_eval_save
from src.ctr_prediction.config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

models = {
    "logistic_regression": LogisticRegression(max_iter=100),
    "lightgbm": LGBMClassifier(),
    "xgboost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_val)[:,1]

    auc = roc_auc_score(y_val, preds)
    loss = log_loss(y_val, preds)

    print(name, auc, loss)
    
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
