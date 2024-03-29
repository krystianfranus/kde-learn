from typing import Optional

import numpy as np
from numpy import ndarray

from .cutils import isdd  # , compute_unbiased_kde

# from scipy.optimize import Bounds, minimize, newton


kernel_properties = {
    "gaussian": (1 / (2 * np.sqrt(np.pi)), 1),
    "uniform": (0.5, 1 / 3),
    "epanechnikov": (0.6, 0.2),
    "cauchy": (5 / (4 * np.pi), 1),
}


def normal_reference(
    x_train: ndarray,
    weights_train: Optional[ndarray] = None,
    kernel_name: str = "gaussian",
) -> ndarray:
    """AMISE-optimal bandwidth for the (assuming) gaussian density.

    See paragraph (3.2.1) in [1].

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points containing data with float type.
    weights_train : ndarray of shape (m_train,), optional
        Weights of data points. If None, all data points are equally weighted.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Returns
    -------
    bandwidth : ndarray of shape (n,)
        Smoothing parameter for scaling the estimator.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> bandwidth = normal_reference(x_train, kernel_name="gaussian")

    References
    ----------
    [1] Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of 'x_train' - should be 2d")
    m_train = x_train.shape[0]

    if weights_train is None:
        weights_train = np.full(m_train, 1 / m_train)
    else:
        if weights_train.ndim != 1:
            raise ValueError("invalid shape of 'weights_train' - should be 1d")
        if weights_train.shape[0] != x_train.shape[0]:
            raise ValueError("invalid size of 'weights_train'")
        if not (weights_train >= 0).all():
            raise ValueError("'weights_train' should be non negative")
        weights_train = weights_train / weights_train.sum()

    if kernel_name not in kernel_properties:
        available_kernels = list(kernel_properties.keys())
        raise ValueError(f"invalid 'kernel_name' - choose one of {available_kernels}")

    m_train = x_train.shape[0]
    # Unbiased weighted standard deviation
    x_mean = np.average(x_train, weights=weights_train, axis=0)
    x_var = np.average((x_train - x_mean) ** 2, weights=weights_train, axis=0)
    weighted_std_x = np.sqrt(m_train / (m_train - 1) * x_var)

    wk, uk = kernel_properties[kernel_name]
    zf = 3 / (8 * np.sqrt(np.pi) * weighted_std_x**5)

    bandwidth = (wk / (uk**2 * zf * m_train)) ** 0.2
    return bandwidth


def direct_plugin(
    x_train: ndarray,
    weights_train: Optional[ndarray] = None,
    kernel_name: str = "gaussian",
    stage: int = 2,
):
    """Direct plug-in method with gaussian kernel used in estimation of integrated
    squared density derivatives limited to maximum value of `stage` equal to 3.

    See paragraph (3.6.1) in [1].

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points containing data with float type.
    weights_train : ndarray of shape (m_train,), optional
        Weights of data points. If None, all data points are equally weighted.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    stage : int, default=2
        Depth of plugging-in (max 3).

    Returns
    -------
    bandwidth : ndarray of shape (n,)
        Smoothing parameter for scaling the estimator.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> bandwidth = direct_plugin(x_train, kernel_name="gaussian", stage=2)

    References
    ----------
    [1] Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of 'x_train' - should be 2d")
    m_train = x_train.shape[0]

    if weights_train is None:
        weights_train = np.full(m_train, 1 / m_train)
    else:
        if weights_train.ndim != 1:
            raise ValueError("invalid shape of 'weights_train' - should be 1d")
        if weights_train.shape[0] != x_train.shape[0]:
            raise ValueError("invalid size of 'weights_train'")
        if not (weights_train >= 0).all():
            raise ValueError("'weights_train' should be non negative")
        weights_train = weights_train / weights_train.sum()

    if kernel_name not in kernel_properties:
        available_kernels = list(kernel_properties.keys())
        raise ValueError(f"invalid 'kernel_name' - choose one of {available_kernels}")

    if not isinstance(stage, int):
        raise ValueError("invalid type of 'stage' - should be of an int type")

    if stage < 0 or stage > 3:
        raise ValueError("invalid 'stage' - should be greater than 0 and less than 4")

    m_train = x_train.shape[0]

    # Unbiased weighted standard deviation
    x_mean = np.average(x_train, weights=weights_train, axis=0)
    x_var = np.average((x_train - x_mean) ** 2, weights=weights_train, axis=0)
    weighted_std_x = np.sqrt(m_train / (m_train - 1) * x_var)
    wk, uk = kernel_properties[kernel_name]

    def _psi(r):
        n = (-1) ** (0.5 * r) * np.math.factorial(r)
        tmp = np.math.factorial(int(0.5 * r)) * np.sqrt(np.pi)
        d = (2 * weighted_std_x) ** (r + 1) * tmp
        return n / d

    def _bw(gd, zf, b):
        # There is hidden uk variable in denominator (equal to 1 for gaussian kernel)
        return (-2 * gd / (zf * m_train)) ** (1 / (b + 1))

    # Gaussian derivatives of order 4, 6 and 8 at zero arg
    gds = {"gd8": 41.888939, "gd6": -5.984134, "gd4": 1.196827}

    r = 2 * stage + 4
    zf = _psi(r)
    while r != 4:
        r -= 2
        gd_at_zero = gds[f"gd{r}"]
        bw = _bw(gd_at_zero, zf, r + 2)
        zf = isdd(x_train, bw, r)

    bandwidth = (wk / (uk**2 * zf * m_train)) ** 0.2
    return bandwidth


# def ste_plugin(
#     x_train: ndarray,
#     weights_train: Optional[ndarray] = None,
#     kernel_name: str = "gaussian",
# ):
#     """Solve-the-equation plug-in method.
#
#     See paragraph (3.6.2) in [1].
#
#     Parameters
#     ----------
#     x_train : ndarray of shape (m_train, n)
#         Data points as an array containing data with float type.
#     weights_train : ndarray of shape (m_train,), optional
#         Weights of data points. If None, all points are equally weighted.
#     kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'  # noqa
#         Name of kernel function.
#
#     Returns
#     -------
#     bandwidth : ndarray of shape (n,)
#         Smoothing parameter.
#
#     Examples
#     --------
#     >>> x_train = np.random.normal(0, 1, size=(100, 1))
#     >>> bandwidth = ste_plugin(x_train, kernel_name="gaussian")
#
#     References
#     ----------
#     [1] Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
#     """
#     if x_train.ndim != 2:
#         raise ValueError("invalid shape of 'x_train' - should be 2d")
#
#     if kernel_name not in kernel_properties:
#         available_kernels = list(kernel_properties.keys())
#         raise ValueError(f"invalid 'kernel_name' - try one of {available_kernels}")
#
#     m_train = x_train.shape[0]
#     if weights_train is None:
#         weights_train = np.full(m_train, 1 / m_train)
#
#     # Unbiased weighted standard deviation
#     x_mean = np.average(x_train, weights=weights_train, axis=0)
#     x_var = np.average((x_train - x_mean) ** 2, weights=weights_train, axis=0)
#     weighted_std_x = np.sqrt(m_train / (m_train - 1) * x_var)
#     wk, uk = kernel_properties[kernel_name]
#
#     def eq(h):
#         a = 0.920 * weighted_std_x * m_train ** (-1 / 7)
#         b = 0.912 * weighted_std_x * m_train ** (-1 / 9)
#         sda = isdd(x_train, weights_train, a, 4)
#         tdb = -isdd(x_train, weights_train, b, 6)
#
#         alpha2 = 1.357 * (sda / tdb) ** (1 / 7) * h ** (5 / 7)
#         sdalpha2 = isdd(x_train, weights_train, alpha2, 4)
#         return (wk / (uk**2 * sdalpha2 * m_train)) ** 0.2 - h
#
#     # Solve the equation using secant method
#     bandwidth0 = normal_reference(x_train, weights_train, kernel_name)
#     bandwidth = newton(eq, bandwidth0)
#     return bandwidth
#
#
# def ml_cv(
#     x_train: ndarray,
#     kernel_name: str = "gaussian",
#     weights_train: Optional[ndarray] = None,
# ):
#     """Likelihood cross-validation.
#
#     See paragraph (3.4.4) in [1].
#
#     Parameters
#     ----------
#     x_train : ndarray of shape (m_train, n)
#         Data points as an array containing data with float type.
#     kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'  # noqa
#         Name of kernel function.
#     weights_train : ndarray of shape (m_train,), optional
#         Weights of data points. If None, all points are equally weighted.
#
#     Returns
#     -------
#     bandwidth : ndarray of shape (n,)
#         Smoothing parameter.
#
#     Examples
#     --------
#     >>> x_train = np.random.normal(0, 1, size=(100, 1))
#     >>> m_train = x_train.shape[0]
#     >>> weights_train = np.full(m_train, 1 / m_train)
#     >>> bandwidth = ml_cv(x_train, "gaussian", weights_train)
#
#     References
#     ----------
#     [1] Silverman, B. W. Density Estimation for Statistics and Data Analysis.
#     Chapman and Hall, 1986.
#     """
#     if x_train.ndim != 2:
#         raise ValueError("invalid shape of 'x_train' - should be 2d")
#
#     if kernel_name not in kernel_properties:
#         available_kernels = list(kernel_properties.keys())
#         raise ValueError(f"invalid 'kernel_name' - try one of {available_kernels}")
#
#     if weights_train is not None:
#         if len(weights_train.shape) != 1:
#             raise ValueError("invalid shape of 'weights_train' - should be 1d")
#         if not (weights_train > 0).all():
#             raise ValueError("'weights_train' must be positive")
#         weights_train = weights_train / weights_train.sum()
#     else:
#         m_train = x_train.shape[0]
#         weights_train = np.full(m_train, 1 / m_train)
#
#     # Minimize the equation with nelder-mead method
#     bandwidth0 = normal_reference(x_train, weights_train, kernel_name)
#     smallest_pos_num = np.nextafter(0, 1)
#
#     def eq(h):
#         scores = compute_unbiased_kde(x_train, weights_train, h, kernel_name)
#         return -np.mean(np.log(scores + smallest_pos_num))
#
#     bounds = Bounds(smallest_pos_num, np.inf)
#     res = minimize(eq, bandwidth0, method="nelder-mead", bounds=bounds)
#     bandwidth = res.x
#     return bandwidth
