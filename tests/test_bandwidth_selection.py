import numpy as np
import pytest

from kdelearn.bandwidth_selection import (
    direct_plugin,
    ml_cv,
    normal_reference,
    ste_plugin,
)


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
def test_normal_reference(x_train, kernel_name):
    bandwidth = normal_reference(x_train, kernel_name)
    assert (bandwidth > 0).all()
    assert x_train.shape[1] == bandwidth.shape[0]


def test_normal_reference_with_invalid_array():
    x_train = np.random.normal(0, 1, size=(100,))
    with pytest.raises(ValueError):
        normal_reference(x_train, "gaussian")


def test_normal_reference_with_invalid_kernel_name(x_train):
    with pytest.raises(ValueError):
        normal_reference(x_train, "abc")


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
def test_direct_plugin(x_train, kernel_name):
    bandwidth = direct_plugin(x_train, kernel_name)
    assert (bandwidth > 0).all()
    assert x_train.shape[1] == bandwidth.shape[0]


def test_direct_plugin_with_invalid_array():
    x_train = np.random.normal(0, 1, size=(100,))
    with pytest.raises(ValueError):
        direct_plugin(x_train, "gaussian")


def test_direct_plugin_with_invalid_kernel_name(x_train):
    with pytest.raises(ValueError):
        direct_plugin(x_train, "abc")


def test_direct_plugin_with_invalid_stage(x_train):
    with pytest.raises(ValueError):
        direct_plugin(x_train, "gaussian", 4)


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
def test_ste_plugin(x_train, kernel_name):
    bandwidth = ste_plugin(x_train, kernel_name)
    assert (bandwidth > 0).all()
    assert x_train.shape[1] == bandwidth.shape[0]


def test_ste_plugin_with_invalid_array():
    x_train = np.random.normal(0, 1, size=(100,))
    with pytest.raises(ValueError):
        ste_plugin(x_train, "gaussian")


def test_ste_plugin_with_invalid_kernel_name(x_train):
    with pytest.raises(ValueError):
        ste_plugin(x_train, "abc")


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
def test_ml_cv(x_train, kernel_name):
    bandwidth = ml_cv(x_train, kernel_name)
    assert (bandwidth > 0).all()
    assert x_train.shape[1] == bandwidth.shape[0]


def test_ml_cv_with_invalid_array():
    x_train = np.random.normal(0, 1, size=(100,))
    with pytest.raises(ValueError):
        ml_cv(x_train, "gaussian")


def test_ml_cv_with_invalid_kernel_name(x_train):
    with pytest.raises(ValueError):
        ml_cv(x_train, "abc")


def test_ml_cv_with_invalid_weights_train(x_train):
    m_train = x_train.shape[0]
    weights_train = np.full((m_train, 2), 1 / m_train)
    with pytest.raises(ValueError):
        ml_cv(x_train, "gaussian", weights_train)

    m_train = x_train.shape[0]
    weights_train = np.full(m_train, -1)
    with pytest.raises(ValueError):
        ml_cv(x_train, "gaussian", weights_train)
