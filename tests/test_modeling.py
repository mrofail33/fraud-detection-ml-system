from fraud_detection.data import clean_target, make_demo_dataset, make_train_test_split, select_features
from fraud_detection.modeling import train_and_evaluate_models


def test_training_returns_three_models_and_metrics():
    df = clean_target(make_demo_dataset(n_rows=1000, random_state=42))
    features = select_features(df)
    X_train, X_test, y_train, y_test = make_train_test_split(
        df, features, test_size=0.25, random_state=42
    )

    models, metrics = train_and_evaluate_models(X_train, X_test, y_train, y_test, random_state=42)

    assert set(models.keys()) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(metrics.columns)
    assert len(metrics) == 3
