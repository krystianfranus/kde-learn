from typing import Optional

import numpy as np
from numpy import ndarray

from kdelearn.cutils import compute_kde

from .utils import scotts_rule


class Kde:
    """
    Kernel density estimator:

    .. math::
        \\hat{f}(x) = \\sum_{i=1}^m w_{i} \\prod_{j=i}^n \\frac{1}{h_j} K(\\frac{x_{j} - x_{i, j}}{h_j})

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> x_train = np.random.normal(0, 1, (1000, 1))
    >>> x_test = np.random.uniform(-1, 1, (10, 1))
    >>> kde = Kde(kernel_name="gaussian").fit(x_train)
    >>> scores = kde.pdf(x_test)

    References
    ----------
    - Silverman, B. W. Density Estimation for Statistics and Data Analysis.
      Boca Raton: Chapman and Hall, 1986.
    """

    def __init__(self, kernel_name: str = "gaussian"):
        self.kernel_name = kernel_name

    def fit(
        self,
        x_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth: Optional[ndarray] = None,
    ):
        """
        Fit kernel density estymator to the data (x_train). This method computes bandwidth.

        Parameters
        ----------
        x_train : `ndarray`
            Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
        weights_train : `ndarray`, optional
            Weights for data points. Must have shape (m_train,). If None is passed, all points get the same weights.
        bandwidth : `ndarray`, optional
            Smoothing parameter. Must have shape (n,).

        Returns
        -------
        self : Kde
            Self instance of `Kde`.

        Examples
        --------
        >>> x_train = np.random.normal(0, 1, size=(1000, 1))
        >>> # with no weights
        >>> kde = Kde(kernel_name="gaussian").fit(x_train, weights_train=None)
        >>> # with weighted data points
        >>> weights_train = np.random.randint(1, 10, size=(1000,))
        >>> kde = Kde(kernel_name="gaussian").fit(x_train, weights_train=weights_train)
        >>> # fixed bandwidth
        >>> bandwidth = np.random.normal(0, 1, size=(1,))
        >>> kde = Kde(kernel_name="gaussian").fit(x_train, bandwidth=bandwidth)
        """
        if len(x_train.shape) != 2:
            raise RuntimeError("x_train must be 2d ndarray")
        self.x_train = np.copy(x_train)

        if weights_train is None:
            m_train = self.x_train.shape[0]
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            if len(weights_train.shape) != 1:
                raise RuntimeError("weights_train must be 1d ndarray")
            if not (weights_train > 0).all():
                raise ValueError("weights_train must be positive")
            self.weights_train = np.copy(weights_train)
            self.weights_train = self.weights_train / self.weights_train.sum()

        if bandwidth is None:
            self.bandwidth = scotts_rule(self.x_train, self.kernel_name)
        else:
            if not (bandwidth > 0).all():
                raise ValueError("bandwidth must be positive")
            self.bandwidth = np.copy(bandwidth)

        return self

    def pdf(self, x_test: ndarray) -> ndarray:
        """
        Compute estimation of probability density function.

        Parameters
        ----------
        x_test : `ndarray`
            Grid data points as a 2D array containing data with `float` type. Must have shape (m_test, n).

        Returns
        -------
        scores : `ndarray`
            Values of kernel density estimator.

        Examples
        --------
        >>> x_train = np.random.normal(0, 1, (1000, 1))
        >>> x_test = np.random.uniform(-1, 1, (10, 1))
        >>> kde = Kde(kernel_name="gaussian").fit(x_train)
        >>> scores = kde.pdf(x_test)
        """
        scores = compute_kde(
            self.x_train,
            x_test,
            self.weights_train,
            self.bandwidth,
            self.kernel_name,
        )
        return scores
