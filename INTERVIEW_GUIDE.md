# Interview Guide: Fraud Detection ML System

## 30-60 Second Explanation

I built a fraud detection machine-learning project that predicts whether a credit card transaction is fraudulent or legitimate. I used a public transaction dataset with anonymized features, cleaned the data, handled missing values, selected numeric features, and used a stratified train/test split to preserve the fraud rate. I trained Logistic Regression, Random Forest, and Gradient Boosting models, then compared them using accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and ROC curves. The main lesson is that accuracy can be misleading for fraud because the data is highly imbalanced, so I focused more on recall, precision, F1, and confusion matrices.

## Plain-Language Project Summary

The goal is to catch suspicious financial transactions. Each row is one transaction. The model looks at transaction features and returns a fraud probability.

The project is intentionally straightforward:

- Load transaction data.
- Clean it.
- Explore the data visually.
- Train a few common classification models.
- Compare the models with the right metrics.
- Save the best model.
- Provide a small command-line prediction interface.

## ML Workflow

1. Data loading
   The code can load `creditcard.csv`, download the OpenML mirror, or run on a small generated demo dataset.

2. Missing-value handling
   Rows without a target label are dropped. Missing feature values are filled with the median value inside each scikit-learn pipeline.

3. Feature selection
   The pipeline uses numeric columns and excludes the target column `Class`.

4. Train/test split
   The split uses stratification so the train and test sets keep a similar fraud-to-legitimate ratio.

5. EDA
   The project creates plots for class balance, transaction amounts, feature correlations, and the most useful correlated features.

6. Model training
   It trains Logistic Regression, Random Forest, and Gradient Boosting.

7. Evaluation
   It compares accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves.

## Why These Models Were Chosen

Logistic Regression:

- Good baseline model.
- Easy to explain.
- Shows how a simple linear classifier performs.

Random Forest:

- Handles non-linear relationships.
- Works well on tabular data.
- Provides a strong comparison against the simpler model.

Gradient Boosting:

- Builds an ensemble of weak learners.
- Often performs well on structured tabular problems.
- Gives another strong model without making the project too complex.

## Metric Explanations

Accuracy:

- Percentage of all predictions that were correct.
- Can be misleading when fraud is rare.

Precision:

- Of the transactions predicted as fraud, how many were actually fraud.
- Useful when false alarms are expensive.

Recall:

- Of the actual fraud cases, how many the model caught.
- Very important in fraud detection because missed fraud can be costly.

F1 score:

- Balance between precision and recall.
- Useful when classes are imbalanced.

ROC-AUC:

- Measures how well the model separates fraud from legitimate transactions across thresholds.

Confusion matrix:

- Shows true positives, true negatives, false positives, and false negatives.
- Helps explain the real business impact of the model.

## Why Accuracy Can Be Misleading

Fraud is rare. If only 0.2% of transactions are fraud, a model could predict every transaction as legitimate and still be about 99.8% accurate. That sounds excellent, but the model caught zero fraud.

That is why this project includes an always-legitimate baseline. It proves that high accuracy alone does not mean the model is useful.

In fraud detection, recall and F1 score usually tell a better story than accuracy alone.

## Likely Interview Questions and Answers

### What problem does this project solve?

It predicts whether a financial transaction is potentially fraudulent using supervised machine learning.

### What is the target variable?

The target is `Class`. A value of `0` means legitimate, and `1` means fraud.

### Why did you use stratified splitting?

The dataset is imbalanced, so a normal random split might put too few fraud cases in either the train or test set. Stratification keeps the class ratio similar in both sets.

### How did you handle missing values?

I dropped rows missing the target because they cannot be used for supervised learning. For feature columns, I used median imputation inside the model pipeline.

### Why is recall important?

Recall measures how many actual fraud cases the model catches. In fraud detection, missing a fraud case can be more costly than investigating a false alarm.

### Why not only optimize accuracy?

Because a model can get very high accuracy by predicting the majority class every time. On imbalanced fraud data, that would miss fraud entirely.

### Which model performed best?

The training script ranks the models by F1 score and saves the best one. The best model can vary depending on whether the demo sample, row-limited data, or full dataset is used.

### What would you improve next?

I would tune classification thresholds, compare precision-recall curves, add cross-validation, try resampling methods like SMOTE, and add model explainability with feature importance or SHAP.

## Resume Bullets

- Built an end-to-end fraud detection ML pipeline using Python, pandas, NumPy, scikit-learn, and matplotlib.
- Cleaned transaction data, handled missing values, selected numeric features, and used stratified train/test splitting for imbalanced classification.
- Trained and evaluated Logistic Regression, Random Forest, and Gradient Boosting models using accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and ROC curves.
- Demonstrated why accuracy is misleading on imbalanced fraud data by comparing trained models against an always-legitimate baseline.
- Organized the project with reusable source modules, setup documentation, automated tests, saved plots, and a lightweight prediction CLI.
