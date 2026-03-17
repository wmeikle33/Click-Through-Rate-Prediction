# Click-Through Rate (CTR) Prediction

A reproducible CTR prediction pipeline for the Kaggle Avazu dataset.  This repo trains a baseline click-through-rate model from `train.csv`, evaluates it on a validation split, and generates a Kaggle-style submission from `test.csv`.  The current baseline uses Logistic Regression with preprocessing for numeric and categorical features. This repository was originally generated from the notebook **Click Through Rate Prediction Final Submission.ipynb** and organized into a Python package + CLI scripts.
You can keep the original notebook under `notebooks/` and iterate on the modular code in `src/` and `scripts/`.

# Summary 

Uses the Avazu CTR Kaggle dataset
Builds a pipeline to:
load and downsample / preprocess high-cardinality categorical features
train a CTR model (LogReg / GBDT / etc.)
evaluate on a validation set using CTR-specific metrics
generate a Kaggle-submittable CSV

## Dataset (Avazu CTR Prediction)

Dataset:
   ~40M training samples
   Highly sparse categorical features
   Binary classification (clicked vs not clicked)
Challenges:
   Extreme class imbalance
   Very high-cardinality categorical variables
   Need for efficient feature encoding

Competition: Avazu Click-Through Rate Prediction (Kaggle)

Download `train.csv` and `test.csv` from the competition page and place them here:

data/raw/train.csv
data/raw/test.csv

## Approach

Pipeline stages:
Data preprocessing
Feature encoding
Model training
Evaluation
Kaggle submission generation

## Feature Engineering

Key feature transformations:
One-hot encoding / hashing for categorical features
Time-based features derived from timestamp
Feature interaction terms

## Modeling

Baseline model:
Logistic Regression (fast, interpretable)
Additional models tested:
Gradient Boosting (LightGBM / XGBoost)
Evaluation metric:
Log Loss
AUC

## Quickstart


This repository was generated from the notebook **Click Through Rate Prediction Final Submission.ipynb** and organized into a Python package + CLI scripts.
You can keep the original notebook under `notebooks/` and iterate on the modular code in `src/` and `scripts/`.

```bash
git clone https://github.com/wmeikle33/Click-Through-Rate-Prediction.git
cd Click-Through-Rate-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data]"
pip install kaggle
python scripts/download_data.py 
python scripts/train.py --csv data/raw/train.csv --label click
python scripts/predict.py --model models/model.joblib --input data/raw/test.csv --output predictions.csv
```

## Project structure

```bash

Click-Through-Rate-Prediction/
├── pyproject.toml
├── pre_commit_config.yaml
├── requirements.txt
├── requirements-dev.txt
├── src/
│   └── ctr_prediction/
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── data.py
├── scripts/
│   ├── train.py
│   └── predict.py
└── tests/


```

## Notes

- The baseline pipeline uses **Logistic Regression** over a simple preprocessing of numeric/categorical columns.
- Swap in gradient-boosting models (XGBoost/LightGBM) if your notebook relied on them; just update `src/model.py`.
- Keep iterating: move stable logic out of the notebook into `src/` functions.

---

# Results

# Reproduce my Score

```

To reproduce the baseline result shown above:

1. Download Avazu `train.csv` and `test.csv` into `data/raw/`
2. Create and activate the virtual environment
3. Install dependencies:
   ```bash
   pip install -e .
   pip install -r requirements.txt

```
