# Fraud Detection ML System

A complete, straightforward internship portfolio project that predicts whether a financial transaction is potentially fraudulent.

The project uses Python, pandas, NumPy, scikit-learn, and matplotlib. It keeps the structure simple enough to explain in an interview while still covering the full machine-learning workflow: data loading, cleaning, EDA, model training, model comparison, evaluation, and a lightweight prediction CLI.

## Dataset

This project is built for the public Credit Card Fraud Detection dataset from ULB / Worldline.

- Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- OpenML mirror: https://www.openml.org/search?type=data&sort=runs&id=1597

The dataset contains anonymized European credit card transactions from September 2013. It has `Time`, `Amount`, PCA-transformed features `V1` through `V28`, and a `Class` target where `1` means fraud and `0` means legitimate. The dataset is highly imbalanced, with fraud representing much less than 1% of all transactions.

Because the full CSV is large, it is not bundled in this repository. You can either download it manually or use the OpenML option.

## Project Structure

```text
fraud-detection-ml-system/
  data/
    README.md
  outputs/
    plots/
    models/
  src/
    fraud_detection/
      data.py
      eda.py
      modeling.py
      predict.py
      train.py
  tests/
  INTERVIEW_GUIDE.md
  README.md
  requirements.txt
  pyproject.toml
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Option 1: Run a Quick Demo

This uses a small generated dataset so the full project can run immediately:

```bash
python -m fraud_detection.train --use-sample
```

This creates:

- `outputs/model_metrics.csv`
- `outputs/model_metrics_with_baseline.csv`
- `outputs/training_summary.json`
- `outputs/models/best_model.joblib`
- plots in `outputs/plots/`

The repo also includes a checked-in sample run under `docs/demo-assets/sample-run/` so reviewers can see example metrics and plots without running the command first:

- `docs/demo-assets/sample-run/model_metrics_with_baseline.csv`
- `docs/demo-assets/sample-run/training_summary.json`
- `docs/demo-assets/sample-run/plots/roc_curves.png`
- `docs/demo-assets/sample-run/plots/random_forest_confusion_matrix.png`

## Option 2: Run With the Real Public Dataset

Download `creditcard.csv` from Kaggle and place it here:

```text
data/creditcard.csv
```

Then run:

```bash
python -m fraud_detection.train --data-path data/creditcard.csv
```

You can also try the OpenML mirror:

```bash
python -m fraud_detection.train --download-openml
```

For a faster first run on a large dataset:

```bash
python -m fraud_detection.train --data-path data/creditcard.csv --max-rows 10000
```

## What the Training Pipeline Does

1. Loads the dataset from CSV, OpenML, or demo sample.
2. Drops rows with missing target labels.
3. Selects numeric features and excludes the target column.
4. Handles missing feature values with median imputation inside each model pipeline.
5. Splits data into train and test sets using stratification so the fraud rate is preserved.
6. Generates EDA plots:
   - fraud vs legitimate transaction counts
   - transaction amount distributions
   - correlation heatmap
   - top features correlated with fraud
7. Trains three models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
8. Compares:
   - accuracy
   - precision
   - recall
   - F1 score
   - ROC-AUC
9. Saves confusion matrices and ROC curves.
10. Saves the best model by F1 score.

## Why Accuracy Is Misleading

Fraud datasets are usually extremely imbalanced. If 99.8% of transactions are legitimate, a model can predict "legitimate" for every transaction and still appear to have about 99.8% accuracy. That model would miss every fraud case, making it useless for fraud detection.

This project demonstrates that issue by saving `outputs/model_metrics_with_baseline.csv`, which includes an "Always Legitimate Baseline." The baseline often has high accuracy but zero recall and zero F1 score because it catches no fraud.

For fraud detection, recall, precision, F1, confusion matrices, and ROC-AUC are more informative than accuracy alone.

## Prediction CLI

After training, run a lightweight prediction:

```bash
python -m fraud_detection.predict --amount 3800 --time 9000
```

You can also pass known feature values as JSON:

```bash
python -m fraud_detection.predict --json "{\"Time\": 9000, \"Amount\": 3800, \"V1\": 0.2, \"V2\": -1.1}"
```

For the real dataset, most features are anonymized PCA values, so a production UI would usually receive all model features from a transaction processing system. This CLI stays intentionally simple.

## Tests

Run:

```bash
pytest
```

The tests use the small generated dataset so they do not require the large public CSV.

## Interview Proof

The repo includes tests, CI, and a quick training smoke test:

- tests: `tests/`
- workflow: `.github/workflows/ci.yml`
- proof notes: `docs/model-results.md`
- sample run artifacts: `docs/demo-assets/sample-run/`

Safe resume wording:

> Built a reusable scikit-learn fraud detection pipeline with data loading, cleaning, stratified splitting, EDA plots, model comparison, imbalanced-class metrics, baseline comparison, model saving, and tests.

## Resume-Ready Bullets

- Built an end-to-end fraud detection machine-learning pipeline using Python, pandas, NumPy, scikit-learn, and matplotlib.
- Performed data cleaning, missing-value handling, feature selection, stratified train/test splitting, and exploratory analysis on imbalanced transaction data.
- Trained and compared Logistic Regression, Random Forest, and Gradient Boosting classifiers using accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and ROC curves.
- Demonstrated why accuracy is misleading for imbalanced fraud detection by comparing models against an always-legitimate baseline.
- Packaged the project with reusable source modules, reproducible setup instructions, saved plots, saved model artifacts, and automated tests.
