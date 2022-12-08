import numpy as np
import pytest


@pytest.fixture(scope="session")
def train_data():
    m_train, n = 100, 1
    x_train = np.random.normal(0, 1, size=(m_train, n))
    weights_train = np.full((m_train,), 1 / m_train)
    return x_train, weights_train


@pytest.fixture(scope="session")
def test_data():
    m_test, n = 100, 1
    x_test = np.random.normal(0, 1, size=(m_test, n))
    return x_test


@pytest.fixture(scope="session")
def classification_data():
    m_train, n = 100, 1
    m_test = m_train

    # Train data
    m_train1 = m_train // 2
    x_train1 = np.random.normal(0, 1, size=(m_train1, n))
    labels_train1 = np.full(m_train1, 1)
    m_train2 = m_train // 2
    x_train2 = np.random.normal(3, 1, size=(m_train2, n))
    labels_train2 = np.full(m_train2, 2)
    x_train = np.concatenate((x_train1, x_train2))
    labels_train = np.concatenate((labels_train1, labels_train2))
    weights_train = np.full((m_train,), 1 / m_train)

    # Test data
    m_test1 = m_test // 2
    x_test1 = np.random.normal(0, 1, size=(m_test1, n))
    labels_test1 = np.full(m_test1, 1)
    m_test2 = m_test // 2
    x_test2 = np.random.normal(3, 1, size=(m_test2, n))
    labels_test2 = np.full(m_test2, 2)
    x_test = np.concatenate((x_test1, x_test2))
    labels_test = np.concatenate((labels_test1, labels_test2))

    return x_train, labels_train, weights_train, x_test, labels_test


@pytest.fixture(scope="session")
def clustering_data():
    m_train, n = 100, 1
    x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
    x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
    x_train = np.concatenate((x_train1, x_train2))
    weights_train = np.full((m_train,), 1 / m_train)
    return x_train, weights_train
