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

    bandwidth = std_x * (wk / (uk ** 2 * zf * m_train)) ** (1 / 5)
    return bandwidth


def estimate_bandwidth_plugin(x_train: ndarray, rank: int):
    """Gaussian kernel specific."""
    if rank > 3:
        raise NotImplementedError("rank can't be greater than 3")
    wk, uk = 1 / (2 * np.sqrt(np.pi)), 1

    m_train = x_train.shape[0]
    std_x = np.std(x_train, axis=0, ddof=1)

    # fmt: off
    def _c(std_x, ksi):
        nominator = np.math.factorial(ksi) * (-1) ** (0.5 * ksi)
        denominator = np.math.factorial(0.5 * ksi) * np.sqrt(np.pi) * (2 * std_x) ** (ksi + 1)
        return nominator / denominator

    def _h(ksi, gaussian_derivative, uk, c, m_train):
        return (-2 * gaussian_derivative(0) / (uk * c * m_train)) ** (1 / (ksi + 1))

    def _C(x_train, ksi, h, gaussian_derivative, m_train):
        return 1 / (m_train ** 2 * h ** (ksi + 1)) * np.sum(np.sum(gaussian_derivative((x_train[:, None] - x_train) / h), axis=0), axis=0)

    def gaussian_derivative_of_order_4(x):
        return 1 / np.sqrt(2 * np.pi) * (x ** 4 - 6 * x ** 2 + 3) * np.exp(-0.5 * x ** 2)

    def gaussian_derivative_of_order_6(x):
        return 1 / np.sqrt(2 * np.pi) * (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * np.exp(-0.5 * x ** 2)

    def gaussian_derivative_of_order_8(x):
        return 1 / np.sqrt(2 * np.pi) * (x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105) * np.exp(-0.5 * x ** 2)
    # fmt: on

    ksi = 2 * rank + 4
    C = _c(std_x, ksi)
    funcs = {
        "func8": gaussian_derivative_of_order_8,
        "func6": gaussian_derivative_of_order_6,
        "func4": gaussian_derivative_of_order_4,
    }
    while ksi != 4:
        gaussian_derivative = funcs[f"func{ksi - 2}"]
        h = _h(ksi, gaussian_derivative, uk, C, m_train)
        C = _C(x_train, ksi - 2, h, gaussian_derivative, m_train)
        ksi -= 2
    zf = C

    bandwidth = (wk / (uk ** 2 * zf * m_train)) ** (1 / 5)
    return bandwidth


def accuracy(labels_true: ndarray, labels_pred: ndarray) -> float:
    """Accuracy score computes fraction of correctly classified samples.

    Parameters
    ----------
    labels_true : `ndarray`
        True (ground truth) labels as a 1D array containing data with `int` type.
    labels_pred : `ndarray`
        Predicted labels returned by a classifier as a 1D array containing data with `int` type.

    Examples
    --------
    >>> labels_true = np.array([0, 1])
    >>> labels_pred = np.array([1, 1])
    >>> accuracy(labels_true, labels_pred)

    Returns
    -------
    accuracy : `float`
        Fraction of correctly classified samples.
    """
    if labels_true.size != labels_pred.size:
        raise RuntimeError("labels_true and labels_pred must have the same size")
    return (labels_true == labels_pred).sum() / labels_true.size
