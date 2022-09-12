import pytest

from kdelearn.kde_funcs import KDEClassification
from kdelearn.metrics import accuracy_loo, pi_kf


def test_accuracy_loo(data_classification):
    x_train, labels_train, _, _ = data_classification
    model = KDEClassification()
    accuracy = accuracy_loo(x_train, labels_train, model)
    assert 0.0 <= accuracy <= 1.0


def test_accuracy_loo_invalid_data(data_classification):
    x_train, labels_train, _, _ = data_classification
    model = KDEClassification()

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        accuracy_loo(x_train_tmp, labels_train, model)

    # Invalid shape of labels_train
    labels_train_tmp = labels_train[:, None]
    with pytest.raises(ValueError):
        accuracy_loo(x_train, labels_train_tmp, model)

    # Invalid dtype of labels_train
    labels_train_tmp = labels_train * 1.0
    with pytest.raises(ValueError):
        accuracy_loo(x_train, labels_train_tmp, model)


def test_accuracy_loo_invalid_model(data_classification):
    x_train, labels_train, _, _ = data_classification
    model = "invalid_model"
    with pytest.raises(AttributeError):
        accuracy_loo(x_train, labels_train, model)


def test_pi_kf(data_outliers_detection):
    x_train, labels_train = data_outliers_detection
    pi = pi_kf(x_train, labels_train)
    assert 0 <= pi <= 1


def test_pi_kf_invalid_data(data_outliers_detection):
    x_train, labels_train = data_outliers_detection

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        pi_kf(x_train_tmp, labels_train)

    # Invalid shape of labels_train
    labels_train_tmp = labels_train[:, None]
    with pytest.raises(ValueError):
        pi_kf(x_train, labels_train_tmp)

    # Invalid dtype of labels_train
    labels_train_tmp = labels_train * 1.0
    with pytest.raises(ValueError):
        pi_kf(x_train, labels_train_tmp)

    # Invalid values in labels_train
    labels_train_tmp = labels_train + 5
    with pytest.raises(ValueError):
        pi_kf(x_train, labels_train_tmp)
