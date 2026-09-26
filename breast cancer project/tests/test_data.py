import numpy as np

from breast_cancer_ml.data import load_data, make_split


def test_dataset_shape_and_positive_class_semantics():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert y.name == "malignant"
    assert set(y.unique()) == {0, 1}
    assert int(y.sum()) == 212  # malignant cases in WDBC


def test_split_is_complete_and_stratified():
    X_train, X_test, y_train, y_test = make_split()
    assert len(X_train) + len(X_test) == 569
    assert X_train.shape[1] == X_test.shape[1] == 30
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_split_is_reproducible():
    split_a = make_split()
    split_b = make_split()
    assert np.array_equal(split_a[1].index, split_b[1].index)
