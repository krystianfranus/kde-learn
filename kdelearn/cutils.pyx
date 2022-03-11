cimport cython

import numpy as np

from libc.math cimport abs, exp, pi, sqrt


cdef double gaussian(double x):
    return 1 / sqrt(2 * pi) * exp(-0.5 * x**2)


cdef double uniform(double x):
    if abs(x) <= 1.0:
        return 0.5
    else:
        return 0.0


cdef double epanechnikov(double x):
    if abs(x) <= 1.0:
        return 3.0 / 4.0 * (1.0 - x**2.0)
    else:
        return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_kde(double[:, :] x_train, double[:] weights_train, double[:] bandwidth, double[:, :] x_test, str kernel_name):
    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n = x_train.shape[1]

    if kernel_name == "gaussian":
        kernel = gaussian
    elif kernel_name == "uniform":
        kernel = uniform
    elif kernel_name == "epanechnikov":
        kernel = epanechnikov
    else:
        raise RuntimeError("invalid kernel name")

    scores = np.zeros(m_test, dtype=np.float64)
    cdef double[:] scores_view = scores

    cdef Py_ssize_t k, i, j
    cdef double tmp

    for k in range(m_test):
        for i in range(m_train):
            tmp = 1.0
            for j in range(n):
                tmp *= 1.0 / bandwidth[j] * kernel((x_test[k, j] - x_train[i, j]) / bandwidth[j])
            scores_view[k] += weights_train[i] * tmp
    return scores
