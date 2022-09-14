import numpy as np
import pytest

from kdelearn.kde_funcs import KDEOutliersDetection


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
@pytest.mark.parametrize(
    "bandwidth_method", ["normal_reference", "direct_plugin", "ste_plugin", "ml_cv"]
)
def test_kde_outliers_detector(x_train, x_test, kernel_name, bandwidth_method):
    outliers_detector = KDEOutliersDetection(kernel_name)
    outliers_detector = outliers_detector.fit(
        x_train,
        bandwidth_method=bandwidth_method,
    )
    labels_pred = outliers_detector.predict(x_test)
    assert outliers_detector.fitted
    assert outliers_detector.threshold > 0
    assert labels_pred.shape[0] == x_test.shape[0]
    assert labels_pred.ndim == 1


def test_kde_outliers_detector_with_weights_train(x_train):
    m_train = x_train.shape[0]
    weights_train = np.ones((m_train,))
    outliers_detector = KDEOutliersDetection()
    outliers_detector.fit(x_train, weights_train=weights_train)
    assert outliers_detector.fitted


def test_kde_outliers_detector_with_fixed_bandwidth(x_train):
    n = x_train.shape[1]
    bandwidth = np.array([0.5] * n)
    outliers_detector = KDEOutliersDetection()
    outliers_detector.fit(x_train, bandwidth=bandwidth)
    assert outliers_detector.fitted


def test_kde_outliers_detector_invalid():
    with pytest.raises(ValueError):
        KDEOutliersDetection("abc")


def test_kde_outliers_detector_fit_invalid(x_train):
    m_train = x_train.shape[0]
    outliers_detector = KDEOutliersDetection()

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        outliers_detector.fit(x_train_tmp)

    # Invalid shape of weights_train
    weights_train = np.ones((m_train, 2))
    with pytest.raises(ValueError):
        outliers_detector.fit(x_train, weights_train=weights_train)

    # Invalid values of weights_train
    weights_train = np.full((m_train,), -1)
    with pytest.raises(ValueError):
        outliers_detector.fit(x_train, weights_train=weights_train)

    # Invalid value of bandwidth_method
    with pytest.raises(ValueError):
        outliers_detector.fit(
            x_train,
            bandwidth_method="abc",
        )

    # Invalid value of r
    r = -0.5
    with pytest.raises(ValueError):
        outliers_detector.fit(x_train, r=r)


def test_kde_outliers_detector_predict_invalid(x_train, x_test):
    outliers_detector = KDEOutliersDetection()

    # No fitting
    with pytest.raises(RuntimeError):
        outliers_detector.predict(x_test)

    # Invalid shape of x_test
    x_test_tmp = x_test.flatten()
    with pytest.raises(ValueError):
        outliers_detector.fit(x_train).predict(x_test_tmp)
