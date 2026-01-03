# Summary 

Uses the Avazu CTR Kaggle dataset
Builds a pipeline to:
load and downsample / preprocess high-cardinality categorical features
train a CTR model (LogReg / GBDT / etc.)
evaluate on a validation set using CTR-specific metrics
generate a Kaggle-submittable CSV

# Click-Through Rate (CTR) Prediction

This repository was generated from the notebook **Click Through Rate Prediction Final Submission.ipynb** and organized into a Python package + CLI scripts.
You can keep the original notebook under `notebooks/` and iterate on the modular code in `src/` and `scripts/`.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Put your training data CSV under data/raw/ or anywhere you like
python scripts/train.py --csv data/raw/train.csv --label <label_column_name>

# Score new samples
python scripts/predict.py --model models/model.joblib --input data/raw/test.csv --output predictions.csv
```

## Project structure

```
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
```

## Notes

- The baseline pipeline uses **Logistic Regression** over a simple preprocessing of numeric/categorical columns.
- Swap in gradient-boosting models (XGBoost/LightGBM) if your notebook relied on them; just update `src/model.py`.
- Keep iterating: move stable logic out of the notebook into `src/` functions.

---

# Notebook Setup

# Introduction and Background

This machine learning problem is titled Click-Through Rate Prediction. The data for this problem comes from a competition posted on the website Kaggle. As per the instructions on the website, the goal of the competition is to find someone who can develop a program that can effectively classify whether something called a click through has occurred on an online advertisement. In analysing online activity, click throughs are defined as when, while surfing the web, someone sees an ad or a link on a webpage and clicks on it as opposed to simply scrolling past. This behaviour indicates that they are interested in the ad and want to get more information about the product or service being offered. Hence, click-through rate (CTR) is a very important metric for evaluating ad performance. As optimizing CTR is of significant importance for advertising companies, click through prediction systems are essential and are commonly used for both sponsored searches and real-time bidding. These systems are often supported by various machine learning algorithms. Thus, through the optimization and continuous evaluation of their machine learning programs, companies continue to search for ways to increase their CTR. Initially, based on the premise of the problem, we understand we are dealing with a binary output variable, 1 meaning click and 0 meaning a non-click.

# Exploratory Data Analysis

Initially, in deciding how best to approach this machine learning problem, properly understanding the structure and nature of the data provided in the various datasets on the Kaggle website was of paramount importance. Specifically, the data came in the form two different data sets, the training data set and the test data set. Initially, I imported both datasets to examine them more closely. After importing them both, I began working with the training dataset to develop the machine learning model to be later employed on the test dataset. In importing th

# Results


