from fraud_detection.data import clean_target, make_demo_dataset, make_train_test_split, select_features


def test_demo_dataset_has_expected_columns():
    df = make_demo_dataset(n_rows=500, random_state=42)

    assert "Amount" in df.columns
    assert "Time" in df.columns
    assert "Class" in df.columns
    assert df["Class"].isin([0, 1]).all()


def test_feature_selection_excludes_target():
    df = clean_target(make_demo_dataset(n_rows=500, random_state=42))
    features = select_features(df)

    assert "Class" not in features
    assert "Amount" in features


def test_stratified_split_preserves_both_classes():
    df = clean_target(make_demo_dataset(n_rows=1000, random_state=42))
    features = select_features(df)
    _, _, y_train, y_test = make_train_test_split(df, features, test_size=0.2, random_state=42)

    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
