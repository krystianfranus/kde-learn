import numpy as np
import pytest

from kdelearn.kde import KDE


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
@pytest.mark.parametrize(
    "bandwidth_method", ["normal_reference", "direct_plugin", "ste_plugin", "ml_cv"]
)
def test_kde(x_train, x_test, kernel_name, bandwidth_method):
    kde = KDE(kernel_name).fit(x_train, bandwidth_method=bandwidth_method)
    scores = kde.pdf(x_test)
    assert kde.fitted
    assert scores.ndim == 1
    assert scores.shape[0] == x_test.shape[0]
    assert scores.all() >= 0


def test_kde_with_invalid_kernel_name(x_train):
    with pytest.raises(ValueError):
        KDE("abc")


def test_kde_with_invalid_x_train():
    x_train = np.array([-1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        KDE("gaussian").fit(x_train)


def test_kde_with_weights_train(x_train):
    m_train = x_train.shape[0]
    weights_train = np.ones((m_train,))
    kde = KDE("gaussian").fit(x_train, weights_train=weights_train)
    assert kde.fitted


def test_kde_with_invalid_weights_train(x_train):
    m_train = x_train.shape[0]
    kde = KDE("gaussian")
    # Invalid shape
    weights_train = np.ones((m_train, 2))
    with pytest.raises(ValueError):
        kde.fit(x_train, weights_train=weights_train)
    # Invalid values
    weights_train = np.full((m_train,), -1)
    with pytest.raises(ValueError):
        kde.fit(x_train, weights_train=weights_train)


def test_kde_with_fixed_bandwidth(x_train):
    n = x_train.shape[1]
    bandwidth = np.array([0.5] * n)
    kde = KDE("gaussian").fit(x_train, bandwidth=bandwidth)
    assert kde.fitted


def test_kde_with_fixed_invalid_bandwidth(x_train):
    n = x_train.shape[1]
    kde = KDE("gaussian")
    # Invalid shape
    bandwidth = np.array([0.5] * n).reshape(n, 1)
    with pytest.raises(ValueError):
        kde.fit(x_train, bandwidth=bandwidth)
    # Invalid values
    bandwidth = np.array([-0.5] * n)
    with pytest.raises(ValueError):
        kde.fit(x_train, bandwidth=bandwidth)
