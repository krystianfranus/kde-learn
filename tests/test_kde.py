import numpy as np
import pytest

from kdelearn.kde import KDE


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
@pytest.mark.parametrize("bandwidth_method", ["normal_reference", "direct_plugin"])
def test_kde(train_data, test_data, kernel_name, bandwidth_method):
    x_train, weights_train = train_data
    x_test = test_data

    kde = KDE(kernel_name).fit(
        x_train,
        weights_train,
        bandwidth_method=bandwidth_method,
    )
    scores = kde.pdf(x_test)
    assert kde.fitted
    assert kde.weights_train.ndim == 1
    assert (kde.weights_train > 0).all()
    assert kde.bandwidth.ndim == 1
    assert (kde.bandwidth > 0).all()
    assert scores.ndim == 1
    assert scores.shape[0] == x_test.shape[0]
    assert (scores >= 0).all()
    assert (scores < 1).all()


def test_kde_with_fixed_bandwidth(train_data):
    x_train, weights_train = train_data
    n = x_train.shape[1]
    bandwidth = np.array([0.5] * n)
    kde = KDE().fit(x_train, weights_train, bandwidth)
    assert kde.fitted
    assert kde.bandwidth.ndim == 1
    assert (kde.bandwidth > 0).all()


def test_kde_with_invalid_kernel_name():
    with pytest.raises(ValueError):
        KDE("abc")


def test_kde_fit_with_invalid_data(train_data):
    x_train, _ = train_data

    kde = KDE()
    m_train, n = x_train.shape

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        kde.fit(x_train_tmp)

    # Invalid shape of weights_train
    weights_train = np.ones((m_train, 2))
    with pytest.raises(ValueError):
        kde.fit(x_train, weights_train)

    # Invalid size of weights_train
    weights_train = np.ones((m_train // 2,))
    with pytest.raises(ValueError):
        kde.fit(x_train, weights_train)

    # Invalid values of weights_train
    weights_train = np.full((m_train,), -1)
    with pytest.raises(ValueError):
        kde.fit(x_train, weights_train)

    # Invalid shape of bandwidth
    bandwidth = np.array([0.5] * n).reshape(n, 1)
    with pytest.raises(ValueError):
        kde.fit(x_train, bandwidth=bandwidth)

    # Invalid size of bandwidth
    bandwidth = np.array([0.5] * (n + 1))
    with pytest.raises(ValueError):
        kde.fit(x_train, bandwidth=bandwidth)

    # Invalid values of bandwidth
    bandwidth = np.array([-0.5] * n)
    with pytest.raises(ValueError):
        kde.fit(x_train, bandwidth=bandwidth)

    # Invalid value of bandwidth_method
    with pytest.raises(ValueError):
        KDE().fit(x_train, bandwidth_method="abc")


def test_kde_pdf_with_invalid_data(train_data, test_data):
    x_train, _ = train_data
    x_test = test_data

    kde = KDE()

    # Invalid shape of x_test
    x_test_tmp = x_test.flatten()
    with pytest.raises(ValueError):
        kde.fit(x_train).pdf(x_test_tmp)

    # Not fitted kde
    with pytest.raises(RuntimeError):
        KDE().pdf(x_test)
