import numpy as np
import pytest

from kdelearn.bandwidth_selection import direct_plugin, normal_reference


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
def test_normal_reference(train_data, kernel_name):
    x_train, weights_train = train_data
    bandwidth = normal_reference(x_train, weights_train, kernel_name)
    assert (bandwidth > 0).all()
    assert x_train.shape[1] == bandwidth.shape[0]


def test_normal_reference_with_invalid_array():
    x_train = np.random.normal(0, 1, size=(100,))
    weights_train = None
    with pytest.raises(ValueError):
        normal_reference(x_train, weights_train, "gaussian")


def test_normal_reference_with_invalid_kernel_name(train_data):
    x_train, weights_train = train_data
    with pytest.raises(ValueError):
        normal_reference(x_train, weights_train, "abc")


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
def test_direct_plugin(train_data, kernel_name):
    x_train, weights_train = train_data
    bandwidth = direct_plugin(x_train, weights_train, kernel_name)
    assert (bandwidth > 0).all()
    assert x_train.shape[1] == bandwidth.shape[0]


def test_direct_plugin_with_invalid_array():
    x_train = np.random.normal(0, 1, size=(100,))
    weights_train = None
    with pytest.raises(ValueError):
        direct_plugin(x_train, weights_train, "gaussian")


def test_direct_plugin_with_invalid_kernel_name(train_data):
    x_train, weights_train = train_data
    with pytest.raises(ValueError):
        direct_plugin(x_train, weights_train, "abc")


def test_direct_plugin_with_invalid_stage(train_data):
    x_train, weights_train = train_data
    with pytest.raises(ValueError):
        direct_plugin(x_train, weights_train, "gaussian", 4)

    with pytest.raises(ValueError):
        direct_plugin(x_train, weights_train, "gaussian", 1.5)
