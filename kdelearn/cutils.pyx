cimport cython

import numpy as np

from libc.math cimport exp, pi, sqrt


cdef double gaussian(double x):
    return 1 / sqrt(2 * pi) * exp(-0.5 * x**2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_kde(double[:, :] x_train, double[:, :] x_test, double[:] bandwidth):
    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n = x_train.shape[1]

    scores = np.zeros(m_test, dtype=np.float64)
    cdef double[:] scores_view = scores

    cdef Py_ssize_t k, i, j
    cdef double tmp

    for k in range(m_test):
        for i in range(m_train):
            tmp = 1.0
            for j in range(n):
                tmp *= gaussian((x_test[k, j] - x_train[i, j]) / bandwidth[j])
            scores_view[k] += tmp
        scores_view[k] *= 1 / (m_train * np.prod(bandwidth))
    return scores
