import pytest

from kdelearn.kde_funcs import KDEClassification
from kdelearn.metrics import accuracy_loo, density_silhouette, pi_kf


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


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
@pytest.mark.parametrize("share_bandwidth", [False, True])
def test_density_silhouette(data_clustering, kernel_name, share_bandwidth):
    x_train, labels_train = data_clustering
    dbs, dbs_mean = density_silhouette(
        x_train, labels_train, kernel_name, share_bandwidth
    )
    assert dbs.shape[0] == x_train.shape[0]
    assert 0 <= dbs_mean <= 1
    assert isinstance(dbs_mean, float)


def test_density_silhouette_invalid_data(data_clustering):
    x_train, labels_train = data_clustering

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        density_silhouette(x_train_tmp, labels_train)

    # Invalid shape of labels_train
    labels_train_tmp = labels_train[:, None]
    with pytest.raises(ValueError):
        density_silhouette(x_train, labels_train_tmp)

    # Invalid dtype of labels_train
    labels_train_tmp = labels_train * 1.0
    with pytest.raises(ValueError):
        density_silhouette(x_train, labels_train_tmp)

    # Invalid values in labels_train
    labels_train_tmp = labels_train + 5
    with pytest.raises(ValueError):
        density_silhouette(x_train, labels_train_tmp)
