import numpy as np
import pytest


@pytest.fixture(scope="session")
def x_train():
    x_train = np.random.normal(0, 1, size=(100, 1))
    return x_train


@pytest.fixture(scope="session")
def x_test():
    x_test = np.random.normal(0, 1, size=(100, 1))
    return x_test
