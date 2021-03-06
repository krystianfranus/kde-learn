from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.optimize import Bounds, minimize, newton

from kdelearn.cutils import compute_unbiased_kde, isdd

kernel_properties = {
    "gaussian": (1 / (2 * np.sqrt(np.pi)), 1),
    "uniform": (0.5, 1 / 3),
    "epanechnikov": (0.6, 0.2),
    "cauchy": (5 / (4 * np.pi), 1),
}


def normal_reference(
    x_train: ndarray,
    kernel_name: str = "gaussian",
) -> ndarray:
    """AMISE-optimal bandwidth for the (assuming) gaussian density.

    See paragraph (3.2.1) in [1].

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points as an array containing data with float type.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Returns
    -------
    bandwidth : ndarray of shape (n,)
        Smoothing parameter.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> bandwidth = normal_reference(x_train, "gaussian")

    References
    ----------
    [1] Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of array - should be two-dimensional")

    if kernel_name not in kernel_properties:
        available_kernels = list(kernel_properties.keys())
        raise ValueError(f"invalid kernel name - try one of {available_kernels}")

    m_train = x_train.shape[0]
    n = x_train.shape[1]
    std_x = np.std(x_train, axis=0, ddof=1)
    wk, uk = kernel_properties[kernel_name]
    zf = n * (n + 2) / (2 ** (n + 2) * np.pi ** (0.5 * n) * std_x**5)

    bandwidth = (wk / (uk**2 * zf * m_train)) ** 0.2
    return bandwidth


def direct_plugin(
    x_train: ndarray,
    kernel_name: str = "gaussian",
    stage: int = 2,
):
    """Direct plug-in method with gaussian kernel used in estimation of integrated
    squared density derivatives limited to max `stage`=3.

    See paragraph (3.6.1) in [1].

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points as an array containing data with float type.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    stage : int, default=2
        Depth of plugging-in (max 3).

    Returns
    -------
    bandwidth : ndarray of shape (n,)
        Smoothing parameter.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> bandwidth = direct_plugin(x_train, "gaussian", 2)

    References
    ----------
    [1] Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of array - should be two-dimensional")

    if kernel_name not in kernel_properties:
        available_kernels = list(kernel_properties.keys())
        raise ValueError(f"invalid kernel name - try one of {available_kernels}")

    if stage < 0 or stage > 3:
        raise ValueError("invalid stage - should be greater than 0 and less than 4")

    m_train = x_train.shape[0]
    n = x_train.shape[1]
    std_x = np.std(x_train, axis=0, ddof=1)
    wk, uk = kernel_properties[kernel_name]

    def _psi(r):
        n = (-1) ** (0.5 * r) * np.math.factorial(r)
        d = (2 * std_x) ** (r + 1) * np.math.factorial(0.5 * r) * np.sqrt(np.pi)
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

    bandwidth = ((n * wk) / (uk**2 * zf * m_train)) ** (1 / (n + 4))
    return bandwidth


def ste_plugin(
    x_train: ndarray,
    kernel_name: str = "gaussian",
):
    """Solve-the-equation plug-in method.

    See paragraph (3.6.2) in [1].

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points as an array containing data with float type.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Returns
    -------
    bandwidth : ndarray of shape (n,)
        Smoothing parameter.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> bandwidth = ste_plugin(x_train, "gaussian")

    References
    ----------
    [1] Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of array - should be two-dimensional")

    if kernel_name not in kernel_properties:
        available_kernels = list(kernel_properties.keys())
        raise ValueError(f"invalid kernel name - try one of {available_kernels}")

    m_train = x_train.shape[0]
    n = x_train.shape[1]
    std_x = np.std(x_train, axis=0, ddof=1)
    wk, uk = kernel_properties[kernel_name]

    def eq(h):
        a = 0.920 * std_x * m_train ** (-1 / 7)
        b = 0.912 * std_x * m_train ** (-1 / 9)
        sda = isdd(x_train, a, 4)
        tdb = -isdd(x_train, b, 6)

        alpha2 = 1.357 * (sda / tdb) ** (1 / 7) * h ** (5 / 7)
        sdalpha2 = isdd(x_train, alpha2, 4)
        return ((n * wk) / (uk**2 * sdalpha2 * m_train)) ** (1 / (n + 4)) - h

    # Solve the equation using secant method
    bandwidth0 = normal_reference(x_train, kernel_name)
    bandwidth = newton(eq, bandwidth0)
    return bandwidth


def ml_cv(
    x_train: ndarray,
    kernel_name: str = "gaussian",
    weights_train: Optional[ndarray] = None,
):
    """Likelihood cross-validation.

    See paragraph (3.4.4) in [1].

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points as an array containing data with float type.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    weights_train : ndarray of shape (m_train,), optional
        Weights of data points. If None, all points are equally weighted.

    Returns
    -------
    bandwidth : ndarray of shape (n,)
        Smoothing parameter.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> m_train = x_train.shape[0]
    >>> weights_train = np.full(m_train, 1 / m_train)
    >>> bandwidth = ml_cv(x_train, "gaussian", weights_train)

    References
    ----------
    [1] Silverman, B. W. Density Estimation for Statistics and Data Analysis.
    Chapman and Hall, 1986.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of array - should be two-dimensional")

    if kernel_name not in kernel_properties:
        available_kernels = list(kernel_properties.keys())
        raise ValueError(f"invalid kernel name - try one of {available_kernels}")

    if weights_train is not None:
        if len(weights_train.shape) != 1:
            raise ValueError("invalid shape of array - should be one-dimensional")
        if not (weights_train > 0).all():
            raise ValueError("array must be positive")
        weights_train = weights_train / weights_train.sum()
    else:
        m_train = x_train.shape[0]
        weights_train = np.full(m_train, 1 / m_train)

    def eq(h):
        scores = compute_unbiased_kde(x_train, weights_train, h, kernel_name)
        return -np.mean(np.log(scores))

    # Minimize the equation with nelder-mead method
    bandwidth0 = normal_reference(x_train, kernel_name)
    smallest_pos_num = np.nextafter(0, 1)
    bounds = Bounds(smallest_pos_num, np.inf)
    res = minimize(eq, bandwidth0, method="nelder-mead", bounds=bounds)
    bandwidth = res.x
    return bandwidth
