import numpy as np


def uniform(x):
    """
    Uniform kernel function.

    Parameters
    ----------
    x : :obj:`ndarray`
        Function argument.

    Returns
    -------
    ndarray
        Values of the function for the given arguments.
    """
    return np.where(np.abs(x) <= 1, 0.5, 0.0)


def gaussian(x):
    """
    Gaussian kernel function.

    Parameters
    ----------
    x : ndarray
        Function argument.

    Returns
    -------
    ndarray
        Values of the function for the given arguments.
    """
    return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)


def epanechnikov(x):
    """
    Epanechnikov kernel function.

    Parameters
    ----------
    x : ndarray
        Function argument.

    Returns
    -------
    ndarray
        Values of the function for the given arguments.
    """
    return np.where(np.abs(x) <= 1, 3 / 4 * (1 - x ** 2), 0.0)


def cauchy(x):
    """
    Cauchy kernel function.

    Parameters
    ----------
    x : ndarray
        Function argument.

    Returns
    -------
    ndarray
        Values of the function for the given arguments.
    """
    return 2 / (np.pi * (x ** 2 + 1) ** 2)
