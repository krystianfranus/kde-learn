import numpy as np
from numpy import ndarray


def scotts_rule(x_train: ndarray, kernel_name: str = "gaussian") -> ndarray:
    """Scott's rule of bandwidth estimation.

    Parameters
    ----------
    x_train : `ndarray`
        Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> bandwidth = scotts_rule(x_train, "gaussian")

    Returns
    -------
    bandwidth : `ndarray`
        Smoothing parameter. Must have shape (n,).
    """
    m_train = x_train.shape[0]
    n = x_train.shape[1]
    std_x = np.std(x_train, axis=0, ddof=1)

    if kernel_name == "gaussian":
        wk, uk = 1 / (2 * np.sqrt(np.pi)), 1
    elif kernel_name == "uniform":
        wk, uk = 1 / 2, 1 / 3
    elif kernel_name == "epanechnikov":
        wk, uk = 0.6, 0.2
    elif kernel_name == "cauchy":
        wk, uk = 5 / (4 * np.pi), 1
    else:
        raise ValueError(f"invalid kernel name: {kernel_name}")
    zf = 3 / (8 * np.sqrt(np.pi))

    bandwidth = std_x * ((n * wk) / (uk ** 2 * zf * m_train)) ** (1 / (n + 4))
    return bandwidth
