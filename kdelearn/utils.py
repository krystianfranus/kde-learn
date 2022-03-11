import numpy as np


def estimate_bandwidth(x_train, kernel_name="gaussian"):
    """
    Bandwidth estimation.

    Parameters
    ----------
    x_train : ndarray of shape (m, n)
        Input.
    kernel : {'uniform', 'gaussian', 'epanechnikov', 'cauchy'}, default='gaussian'
        Kernel name.

    Returns
    -------
    bandwidth : array_like of shape (n,)
        Bandwidth (smoothing) parameter.
    """
    m_train = x_train.shape[0]
    std_x = np.std(x_train, axis=0, ddof=1)
    if m_train == 1:
        std_x = np.std(x_train, axis=0)
    std_x[std_x == 0] = 1

    if kernel_name == "gaussian":
        W, U = 1 / (2 * np.sqrt(np.pi)), 1
    elif kernel_name == "uniform":
        W, U = 1 / 2, 1 / 3
    elif kernel_name == "cauchy":
        W, U = 5 / (4 * np.pi), 1
    elif kernel_name == "epanechnikov":
        W, U = 0.6, 0.2
    else:
        raise ValueError(f"invalid kernel: {kernel_name}")
    WU2 = W / (U ** 2)
    Z = 3 / (8 * np.sqrt(np.pi))

    bandwidth = (WU2 / (Z * m_train)) ** (1 / 5) * std_x
    return bandwidth
