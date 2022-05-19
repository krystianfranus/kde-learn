cimport cython

import numpy as np
from numpy import ndarray

from libc.math cimport abs, exp, pi, sqrt


cdef double cgaussian(double x):
    return 1.0 / sqrt(2.0 * pi) * exp(-0.5 * x ** 2)


# wrapper to expose cgaussian() to Python
def gaussian(x: float) -> float:
    """Gaussian kernel function.

    .. math::
        K(x) = \\frac{1}{\\sqrt{2 \\pi}} \\exp \\left( \\frac{x^2}{2} \\right)
    """
    return cgaussian(x)


cdef double cuniform(double x):
    if abs(x) <= 1.0:
        return 0.5
    else:
        return 0.0


# wrapper to expose cuniform() to Python
def uniform(x: float) -> float:
    """Uniform kernel function.

    .. math::
        K(x) = 0.5 \\quad \\text{if } |x| \\leq 0 \\quad  \\text{else } 0
    """
    return cuniform(x)


cdef double cepanechnikov(double x):
    if abs(x) <= 1.0:
        return 3.0 / 4.0 * (1.0 - x ** 2.0)
    else:
        return 0.0


# wrapper to expose cepanechnikov() to Python
def epanechnikov(x: float) -> float:
    """Epanechnikov kernel function.

    .. math::
        K(x) = \\frac{3}{4} (1-x^2) \\quad \\text{if } |x| \\leq 0 \\quad  \\text{else } 0
    """
    return cepanechnikov(x)


cdef double ccauchy(double x):
    return 2.0 / (pi * (x ** 2 + 1.0) ** 2)


# wrapper to expose ccauchy() to Python
def cauchy(x: float) -> float:
    """Cauchy kernel function.

    .. math::
        K(x) = \\frac{2}{\\pi (x^2 + 1)^2}
    """
    return ccauchy(x)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_kde(
    double[:, :] x_train,
    double[:, :] x_test,
    double[:] weights_train,
    double[:] bandwidth,
    str kernel_name,
):
    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n = x_train.shape[1]

    if kernel_name == "gaussian":
        kernel = cgaussian
    elif kernel_name == "uniform":
        kernel = cuniform
    elif kernel_name == "epanechnikov":
        kernel = cepanechnikov
    elif kernel_name == "cauchy":
        kernel = ccauchy
    else:
        raise ValueError("invalid kernel name")

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ckde(
    double[:, :] x_train,
    double[:, :] w_train,
    double[:, :] x_test,
    double[:, :] w_test,
    double[:] weights_train,
    double[:] bandwidth_x,
    double[:] bandwidth_w,
    str kernel_name,
):
    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n_x = x_train.shape[1]
    cdef Py_ssize_t n_w = w_train.shape[1]

    if kernel_name == "gaussian":
        kernel = cgaussian
    elif kernel_name == "uniform":
        kernel = cuniform
    elif kernel_name == "epanechnikov":
        kernel = cepanechnikov
    elif kernel_name == "cauchy":
        kernel = ccauchy
    else:
        raise ValueError("invalid kernel name")

    scores = np.zeros(m_test, dtype=np.float64)
    cdef double[:] scores_view = scores

    cdef Py_ssize_t k, i, j
    cdef double tmp, scores_x, scores_w

    for k in range(m_test):
        scores_z, scores_w = 0.0, 0.0
        for i in range(m_train):
            tmp = 1.0
            for j in range(n_x):
                tmp *= 1.0 / bandwidth_x[j] * kernel((x_test[k, j] - x_train[i, j]) / bandwidth_x[j])
            for j in range(n_w):
                tmp *= 1.0 / bandwidth_w[j] * kernel((w_test[k, j] - w_train[i, j]) / bandwidth_w[j])
            scores_z += weights_train[i] * tmp

        for i in range(m_train):
            tmp = 1.0
            for j in range(n_w):
                tmp *= 1.0 / bandwidth_w[j] * kernel((w_test[k, j] - w_train[i, j]) / bandwidth_w[j])
            scores_w += weights_train[i] * tmp
        scores_view[k] = scores_z / scores_w
    return scores
