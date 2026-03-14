<<<<<<< HEAD
# Click-Through Rate (CTR) Prediction

A reproducible CTR prediction pipeline for the Kaggle Avazu dataset.  This repo trains a baseline click-through-rate model from `train.csv`, evaluates it on a validation split, and generates a Kaggle-style submission from `test.csv`.  The current baseline uses Logistic Regression with preprocessing for numeric and categorical features. This repository was originally generated from the notebook **Click Through Rate Prediction Final Submission.ipynb** and organized into a Python package + CLI scripts.
You can keep the original notebook under `notebooks/` and iterate on the modular code in `src/` and `scripts/`.

=======
>>>>>>> 2335b48 (New commit)
# Summary 

Uses the Avazu CTR Kaggle dataset
Builds a pipeline to:
load and downsample / preprocess high-cardinality categorical features
train a CTR model (LogReg / GBDT / etc.)
evaluate on a validation set using CTR-specific metrics
generate a Kaggle-submittable CSV

## Dataset (Avazu CTR Prediction)

Competition: Avazu Click-Through Rate Prediction (Kaggle)

Download `train.csv` and `test.csv` from the competition page and place them here:

data/raw/train.csv
data/raw/test.csv

<<<<<<< HEAD
## Quickstart

```bash
git clone https://github.com/wmeikle33/Click-Through-Rate-Prediction.git
cd Click-Through-Rate-Prediction
python -m venv .venv
source .venv/bin/activate
pip install -e .
=======
# Click-Through Rate (CTR) Prediction

This repository was generated from the notebook **Click Through Rate Prediction Final Submission.ipynb** and organized into a Python package + CLI scripts.
You can keep the original notebook under `notebooks/` and iterate on the modular code in `src/` and `scripts/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
>>>>>>> 2335b48 (New commit)
pip install -r requirements.txt
python scripts/train.py --csv data/raw/train.csv --label click
python scripts/predict.py --model models/model.joblib --input data/raw/test.csv --output predictions.csv
```

## Project structure

```
<<<<<<< HEAD
Click-Through-Rate-Prediction/
├── pyproject.toml
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

=======
ctr-prediction/
├── src/                   # reusable code (data, features, model)
├── scripts/               # CLI entrypoints: train/predict
├── notebooks/             # original notebook + exported .py
├── data/raw/              # place raw data here (gitignored)
├── models/                # saved models (gitignored)
├── reports/figures/       # plots (gitignored)
├── tests/                 # add unit tests if needed
├── requirements.txt
└── README.md
>>>>>>> 2335b48 (New commit)
```

## Notes

- The baseline pipeline uses **Logistic Regression** over a simple preprocessing of numeric/categorical columns.
- Swap in gradient-boosting models (XGBoost/LightGBM) if your notebook relied on them; just update `src/model.py`.
- Keep iterating: move stable logic out of the notebook into `src/` functions.

---

# Results

# Reproduce my Score

<<<<<<< HEAD
```

## Reproduce my score

To reproduce the baseline result shown above:

1. Download Avazu `train.csv` and `test.csv` into `data/raw/`
2. Create and activate the virtual environment
3. Install dependencies:
   ```bash
   pip install -e .
   pip install -r requirements.txt

```

=======
>>>>>>> 2335b48 (New commit)

