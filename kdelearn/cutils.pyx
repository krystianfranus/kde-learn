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
    double[:] weights_train,
    double[:, :] x_test,
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
        raise ValueError("invalid 'kernel_name'")

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
def compute_d(
    double[:, :] w_train,
    double[:] weights_train,
    double[:] w_star,
    double[:] bandwidth_w,
    str kernel_name,
):
    cdef Py_ssize_t m_train = w_train.shape[0]
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
        raise ValueError("invalid 'kernel_name'")

    d = np.zeros((m_train,), dtype=np.float64)
    cdef double[:] d_view = d

    cdef Py_ssize_t i, j
    cdef double tmp, d_sum

    d_sum = 0.0
    for i in range(m_train):
        tmp = 1.0
        for j in range(n_w):
            tmp *= kernel((w_star[j] - w_train[i, j]) / bandwidth_w[j])
        d_view[i] = weights_train[i] * tmp
        d_sum += d_view[i]

    for i in range(m_train):
        d_view[i] = d_view[i] / d_sum
    return d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_ckde(
    double[:, :] x_train,
    double[:] d,
    double[:, :] x_test,
    double[:] bandwidth_x,
    str kernel_name,
):
    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n_x = x_train.shape[1]

    if kernel_name == "gaussian":
        kernel = cgaussian
    elif kernel_name == "uniform":
        kernel = cuniform
    elif kernel_name == "epanechnikov":
        kernel = cepanechnikov
    elif kernel_name == "cauchy":
        kernel = ccauchy
    else:
        raise ValueError("invalid 'kernel_name'")

    scores = np.zeros(m_test, dtype=np.float64)
    cdef double[:] scores_view = scores

    cdef Py_ssize_t k, i, j
    cdef double tmp

    for k in range(m_test):
        for i in range(m_train):
            tmp = 1.0
            for j in range(n_x):
                tmp *= 1.0 / bandwidth_x[j] * kernel((x_test[k, j] - x_train[i, j]) / bandwidth_x[j])
            scores_view[k] += d[i] * tmp
    return scores


cdef double cgd4(double x):
    """Gaussian derivative of order 4."""
    return (x ** 4 - 6 * x ** 2 + 3) * gaussian(x)


cdef double cgd6(double x):
    """Gaussian derivative of order 6."""
    return (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * gaussian(x)


cdef double cgd8(double x):
    """Gaussian derivative of order 8."""
    return (x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105) * gaussian(x)


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_unbiased_kde(
    double[:, :] x_train,
    double[:] weights_train,
    double[:] bandwidth,
    str kernel_name,
):
    cdef Py_ssize_t m_train = x_train.shape[0]
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
        raise ValueError("invalid 'kernel_name'")

    scores = np.zeros(m_train, dtype=np.float64)
    cdef double[:] scores_view = scores

    cdef Py_ssize_t k, i, j
    cdef double tmp

    for k in range(m_train):
        for i in range(m_train):
            if k == i:
                continue
            tmp = 1.0
            for j in range(n):
                tmp *= 1.0 / bandwidth[j] * kernel((x_train[k, j] - x_train[i, j]) / bandwidth[j])
            scores_view[k] += weights_train[i] * tmp
    return scores


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gradient_ascent(
    double[:, :] x_train,
    double[:] weights_train,
    double[:, :] x_test,
    double[:] bandwidth,
    double epsilon,
):
    if epsilon <= 0:
        raise ValueError("invalid value of 'epsilon' - should be positive")

    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n = x_train.shape[1]

    x_k = np.copy(x_test)
    cdef double[:, :] x_k_view = x_k
    cdef double[:] x_k_prev_view

    cdef Py_ssize_t i1, j1, i2, j2
    cdef double numerator, denominator, tmp, dist

    for i1 in range(m_train):
        while True:
            x_k_prev_view = np.copy(x_k_view[i1])

            for j1 in range(n):
                numerator = 0.0
                denominator = 0.0
                for i2 in range(m_train):
                    tmp = 0.0
                    for j2 in range(n):
                        tmp += ((x_k_view[i1, j2] - x_train[i2, j2]) / bandwidth[j2]) ** 2
                    tmp = exp(-0.5 * tmp)
                    numerator += - weights_train[i2] * (x_k_view[i1, j1] - x_train[i2, j1]) * tmp
                    denominator += weights_train[i2] * tmp
                x_k_view[i1, j1] += 1.0 / (n + 2.0) * numerator / denominator

            dist = 0.0
            for j1 in range(n):
                dist += (x_k_view[i1, j1] - x_k_prev_view[j1]) ** 2
            dist = sqrt(dist)
            if dist < epsilon:
                break
    return x_k


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mean_shift(
    double[:, :] x_train,
    double[:] weights_train,
    double[:, :] x_test,
    double[:] bandwidth,
    double epsilon,
):
    if epsilon <= 0:
        raise ValueError("invalid value of 'epsilon' - should be positive")

    cdef Py_ssize_t m_train = x_train.shape[0]
    cdef Py_ssize_t m_test = x_test.shape[0]
    cdef Py_ssize_t n = x_train.shape[1]

    x_k = np.copy(x_test)
    cdef double[:, :] x_k_view = x_k
    cdef double[:] x_k_prev_view

    cdef Py_ssize_t i1, j1, i2, j2
    cdef double numerator, denominator, tmp, dist

    for i1 in range(m_test):
        while True:
            x_k_prev_view = np.copy(x_k_view[i1])

            for j1 in range(n):
                numerator = 0.0
                denominator = 0.0
                for i2 in range(m_train):
                    tmp = 1.0
                    for j2 in range(n):
                        tmp *= cgaussian((x_k_view[i1, j2] - x_train[i2, j2]) / bandwidth[j2]) / bandwidth[j2]
                    numerator += weights_train[i2] * tmp * x_train[i2, j1]
                    denominator += weights_train[i2] * tmp
                x_k_view[i1, j1] = numerator / denominator

            dist = 0.0
            for j1 in range(n):
                dist += (x_k_view[i1, j1] - x_k_prev_view[j1]) ** 2
            dist = sqrt(dist)
            if dist < epsilon:
                break
    return x_k


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def assign_labels(
    double[:, :] x_k,
    double delta,
):
    if delta <= 0:
        raise ValueError("invalid value of 'delta' - should be positive")

    cdef Py_ssize_t m_train = x_k.shape[0]
    cdef Py_ssize_t n = x_k.shape[1]

    cdef double[:, :] x_rep_view = np.copy(x_k[0:1])
    labels = np.zeros(m_train, np.int32)
    cdef int[:] labels_view = labels

    cdef Py_ssize_t i, r, j
    cdef double dist
    cdef bint add_new_rep

    for i in range(1, m_train):
        rep_size = x_rep_view.shape[0]
        add_new_rep = True

        for r in range(rep_size):
            dist = 0.0
            for j in range(n):
                dist += (x_k[i, j] - x_rep_view[r, j]) ** 2
            dist = sqrt(dist)
            if dist < delta:
                labels_view[i] = r
                add_new_rep = False
                break

        if add_new_rep:
            x_rep_view = np.append(x_rep_view, x_k[i:i+1], axis=0)
            labels_view[i] = rep_size
    return labels
