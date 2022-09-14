import numpy as np
import pytest

from kdelearn.kde_funcs import KDEClustering


@pytest.mark.parametrize("algorithm", ["gradient_ascent", "mean_shift"])
@pytest.mark.parametrize(
    "bandwidth_method", ["normal_reference", "direct_plugin", "ste_plugin", "ml_cv"]
)
def test_kde_clustering(data_clustering, bandwidth_method, algorithm):
    x_train, _ = data_clustering
    clustering = KDEClustering()
    clustering = clustering.fit(
        x_train,
        bandwidth_method=bandwidth_method,
    )
    labels_pred = clustering.predict(algorithm)
    assert clustering.fitted
    assert clustering.bandwidth.ndim == 1
    assert (clustering.bandwidth > 0).all()
    assert labels_pred.shape[0] == x_train.shape[0]
    assert labels_pred.ndim == 1


def test_kde_clustering_with_fixed_bandwidth(data_clustering):
    x_train, _ = data_clustering
    n = x_train.shape[1]
    bandwidth = np.array([0.5] * n)
    clustering = KDEClustering()
    clustering.fit(x_train, bandwidth=bandwidth)
    assert clustering.fitted
    assert clustering.bandwidth.ndim == 1
    assert (clustering.bandwidth > 0).all()


def test_kde_clustering_fit_invalid(data_clustering):
    x_train, _ = data_clustering
    n = x_train.shape[1]
    clustering = KDEClustering()

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        clustering.fit(x_train_tmp)

    # Invalid shape of bandwidth
    bandwidth = np.array([0.5] * n).reshape(n, 1)
    with pytest.raises(ValueError):
        clustering.fit(x_train, bandwidth=bandwidth)

    # Invalid size of bandwidth
    bandwidth = np.array([0.5] * (n + 1))
    with pytest.raises(ValueError):
        clustering.fit(x_train, bandwidth=bandwidth)

    # Invalid values of bandwidth
    bandwidth = np.array([-0.5] * n)
    with pytest.raises(ValueError):
        clustering.fit(x_train, bandwidth=bandwidth)

    # Invalid value of bandwidth_method
    with pytest.raises(ValueError):
        clustering.fit(
            x_train,
            bandwidth_method="abc",
        )


def test_kde_clustering_predict_invalid(data_clustering):
    x_train, _ = data_clustering
    clustering = KDEClustering()

    # No fitting
    with pytest.raises(RuntimeError):
        clustering.predict("abc")

    # Invalid value of algorithm
    with pytest.raises(ValueError):
        KDEClustering().fit(x_train).predict("abc")
