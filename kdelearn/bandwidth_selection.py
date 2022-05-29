import numpy as np
from numpy import ndarray
from scipy.optimize import newton

from kdelearn.cutils import isdd


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
    zf = 3 / (8 * np.sqrt(np.pi) * std_x ** 5)

    bandwidth = (wk / (uk ** 2 * zf * m_train)) ** 0.2
    return bandwidth


def direct_plugin(x_train: ndarray, kernel_name: str = "gaussian", stage: int = 2):
    """Direct plug-in method with gaussian kernel used in estimation of integrated squared density derivatives
    limited to max stage=3. See paragraph (3.6.1) in the Wand's book.

    Parameters
    ----------
    x_train : `ndarray`
        Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    stage : `int`, default=2
        Depth of plugging-in. Max 3.

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
    if stage < 0 or stage > 3:
        raise ValueError("stage must be greater than 0 and less than 4")

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

    bandwidth = (wk / (uk ** 2 * zf * m_train)) ** 0.2
    return bandwidth


def ste_plugin(x_train: ndarray, kernel_name: str = "gaussian"):
    """Solve-the-equation plug-in method. See paragraph (3.6.2) in the Wand's book.

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
    >>> x_train = np.random.normal(0, 1, size=(1000, 1))
    >>> bandwidth = ste_plugin(x_train, "gaussian")

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

    def eq(h):
        a = 0.920 * std_x * m_train ** (-1 / 7)
        b = 0.912 * std_x * m_train ** (-1 / 9)
        sda = isdd(x_train, a, 4)
        tdb = -isdd(x_train, b, 6)

        alpha2 = 1.357 * (sda / tdb) ** (1 / 7) * h ** (5 / 7)
        sdalpha2 = isdd(x_train, alpha2, 4)
        return (wk / (uk ** 2 * sdalpha2 * m_train)) ** 0.2 - h

    # Solve the equation using secant method
    bandwidth0 = normal_reference(x_train, kernel_name)
    bandwidth = newton(eq, bandwidth0)
    return bandwidth
