cimport cython

import numpy as np
from numpy import ndarray

from libc.math cimport abs, exp, pi, sqrt
from .cutils import gaussian


cdef double cgd4(double x):
    """Gaussian derivative of order 4."""
    return (x ** 4 - 6 * x ** 2 + 3) * gaussian(x)


# wrapper to expose cgd4() to Python
def gd4(x):
    return cgd4(x)


cdef double cgd6(double x):
    """Gaussian derivative of order 6."""
    return (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * gaussian(x)


# wrapper to expose cgd6() to Python
def gd6(x):
    return cgd6(x)


cdef double cgd8(double x):
    """Gaussian derivative of order 8."""
    return (x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105) * gaussian(x)


# wrapper to expose cgd8() to Python
def gd8(x):
    return cgd8(x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def isdd(
    double[:, :] x_train,
    double[:] bandwidth,
    int r,
):
    """Estimation of integrated squared density derivative"""
    if r == 4:
        func = cgd4
    elif r == 6:
        func = cgd6
    elif r == 8:
        func = cgd8

    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t n = x_train.shape[1]
    cdef Py_ssize_t j, i1, i2

    result = np.zeros(n, dtype=np.float64)
    cdef double[:] result_view = result

    for j in range(n):
        for i1 in range(m_train):
            for i2 in range(m_train):
                result_view[j] += func((x_train[i1, j] - x_train[i2, j]) / bandwidth[j])
        result_view[j] *= 1 / (m_train ** 2 * bandwidth[j] ** (r + 1))
    return result
