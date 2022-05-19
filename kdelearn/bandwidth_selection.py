import numpy as np
from numpy import ndarray

from kdelearn.cbw_selection import gd4, gd6, gd8, isdd


def normal_reference(x_train: ndarray, kernel_name: str = "gaussian") -> ndarray:
    """AMISE-optimal bandwidth for the (assuming) gaussian density. See paragraph (3.2.1) in the Wand's book.

    Parameters
    ----------
    x_train : `ndarray`
        Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Returns
    -------
    bandwidth : `ndarray`
        Smoothing parameter. Must have shape (n,).

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> bandwidth = normal_reference(x_train, "gaussian")

    References
    ----------
    - Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    m_train = x_train.shape[0]
    std_x = np.std(x_train, axis=0, ddof=1)

    if kernel_name == "gaussian":
        wk, uk = 1 / (2 * np.sqrt(np.pi)), 1
    elif kernel_name == "uniform":
        wk, uk = 0.5, 1 / 3
    elif kernel_name == "epanechnikov":
        wk, uk = 0.6, 0.2
    elif kernel_name == "cauchy":
        wk, uk = 5 / (4 * np.pi), 1
    else:
        raise ValueError(f"invalid kernel name: {kernel_name}")
    zf = 3 / (8 * np.sqrt(np.pi))

    bandwidth = std_x * (wk / (uk ** 2 * zf * m_train)) ** 0.2
    return bandwidth


def direct_plugin(x_train: ndarray, kernel_name: str = "gaussian", stage: int = 2):
    """Direct plug-in method with gaussian kernel used in estimator of integrated squared density derivatives
    limited to max stage=3. See paragraph (3.6.1) in the Wand's book.

    Parameters
    ----------
    x_train : `ndarray`
        Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    stage : `int`, default=2
        Depth of plugging-in.

    Returns
    -------
    bandwidth : `ndarray`
        Smoothing parameter. Must have shape (n,).

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> bandwidth = direct_plugin(x_train, "gaussian", 2)

    References
    ----------
    - Wand, M. P. and Jones, M. C. Kernel Smoothing. Chapman and Hall, 1995.
    """
    if stage > 3:
        raise ValueError("stage must be less or equal 3")

    m_train = x_train.shape[0]
    std_x = np.std(x_train, axis=0, ddof=1)

    if kernel_name == "gaussian":
        wk, uk = 1 / (2 * np.sqrt(np.pi)), 1
    elif kernel_name == "uniform":
        wk, uk = 0.5, 1 / 3
    elif kernel_name == "epanechnikov":
        wk, uk = 0.6, 0.2
    elif kernel_name == "cauchy":
        wk, uk = 5 / (4 * np.pi), 1
    else:
        raise ValueError(f"invalid kernel name: {kernel_name}")

    def _psi(r):  # ok
        n = (-1) ** (0.5 * r) * np.math.factorial(r)
        d = (2 * std_x) ** (r + 1) * np.math.factorial(0.5 * r) * np.sqrt(np.pi)
        return n / d

    def _bw(gd, zf, b):  # ok
        # there is hidden uk variable in denominator (equal to 1 for gaussian kernel)
        return (-2 * gd(0) / (zf * m_train)) ** (1 / (b + 1))

    gds = {"gd8": gd8, "gd6": gd6, "gd4": gd4}

    r = 2 * stage + 4
    zf = _psi(r)
    while r != 4:
        r -= 2
        der = gds[f"gd{r}"]
        bw = _bw(der, zf, r + 2)
        zf = isdd(x_train, bw, r)

    bandwidth = (wk / (uk ** 2 * zf * m_train)) ** 0.2
    return bandwidth
