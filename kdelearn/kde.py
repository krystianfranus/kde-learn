from __future__ import annotations  # Needed in type annotation of return in fit method

from typing import Optional

import numpy as np
from numpy import ndarray

from .bandwidth_selection import direct_plugin, kernel_properties, normal_reference
from .cutils import compute_kde


class KDE:
    """Kernel density estimator with product kernel:

    .. math::
        \\hat{f}(x) = \\sum_{i=1}^m w_{i} \\prod_{j=i}^n \\frac{1}{h_j}
        K \\left( \\frac{x_{j} - x_{i, j}}{h_j} \\right), \\quad x \\in \\mathbb{R}^n

    Read more :ref:`here <unconditional_kde>`.

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data
    >>> m_train, n = 100, 1
    >>> x_train = np.random.normal(0, 1, size=(m_train, n))
    >>> # Fit
    >>> kde = KDE("gaussian").fit(x_train)

    References
    ----------
    [1] Silverman, B. W. Density Estimation for Statistics and Data Analysis.
    Chapman and Hall, 1986.

    [2] Wand, M. P., Jones M.C. Kernel Smoothing. Chapman and Hall, 1995.
    """

    def __init__(self, kernel_name: str = "gaussian") -> None:
        if kernel_name not in kernel_properties:
            available_kernels = list(kernel_properties.keys())
            raise ValueError(f"invalid 'kernel_name' - try one of {available_kernels}")
        self.kernel_name = kernel_name
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth: Optional[ndarray] = None,
        bandwidth_method: str = "normal_reference",
        **kwargs,
    ) -> KDE:
        """Fit the estimator.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n)
            Array containing data points with float type for constructing the
            estimator.
        weights_train : ndarray of shape (m_train,), optional
            Weights of data points. If None, all data points are equally weighted.
        bandwidth : ndarray of shape (n,), optional
            Smoothing parameter for scaling the estimator. If None, `bandwidth_method`
            is used to compute the `bandwidth`.
        bandwidth_method : {'normal_reference', 'direct_plugin'}, \
                default='normal_reference'
            Name of bandwidth selection method used to compute `bandwidth` when it is
            not given explicitly.

        Returns
        -------
        self : object
            Fitted self instance of KDE.

        Examples
        --------
        >>> # Prepare data
        >>> m_train, n = 100, 1
        >>> x_train = np.random.normal(0, 1, size=(m_train, n))
        >>> weights_train = np.full((m_train,), 1 / m_train)
        >>> bandwidth = np.full((n,), 1.0)
        >>> # Fit the estimator
        >>> kde = KDE().fit(x_train, weights_train, bandwidth)
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of 'x_train' - should be 2d")
        self.x_train = x_train
        m_train, n = self.x_train.shape

        if weights_train is None:
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            if weights_train.ndim != 1:
                raise ValueError("invalid shape of 'weights_train' - should be 1d")
            if weights_train.shape[0] != x_train.shape[0]:
                raise ValueError("invalid size of 'weights_train'")
            if not (weights_train >= 0).all():
                raise ValueError("'weights_train' should be non negative")
            self.weights_train = weights_train / weights_train.sum()

        if bandwidth is None:
            if bandwidth_method == "normal_reference":
                self.bandwidth = normal_reference(
                    self.x_train,
                    self.weights_train,
                    self.kernel_name,
                )
            elif bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                self.bandwidth = direct_plugin(
                    self.x_train,
                    self.weights_train,
                    self.kernel_name,
                    stage,
                )
            else:
                raise ValueError("invalid 'bandwidth_method'")
        else:
            if bandwidth.ndim != 1:
                raise ValueError("invalid shape of 'bandwidth' - should be 1d")
            if bandwidth.shape[0] != n:
                raise ValueError(
                    f"invalid size of 'bandwidth' - should contain {n} values"
                )
            if not (bandwidth > 0).all():
                raise ValueError("'bandwidth' should be positive")
            self.bandwidth = bandwidth

        self.fitted = True
        return self

    def pdf(self, x_test: ndarray) -> ndarray:
        """Compute probability density.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Argument of the estimator - array containing data points with float type.

        Returns
        -------
        scores : ndarray of shape (m_test,)
            Computed estimation of probability densities for testing data points
            `x_test`.

        Examples
        --------
        >>> # Prepare data
        >>> m_train, n = 100, 1
        >>> m_test = 10
        >>> x_train = np.random.normal(0, 1, (m_train, n))
        >>> x_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        >>> # Fit the estimator
        >>> kde = KDE().fit(x_train)
        >>> # Compute pdf
        >>> scores = kde.pdf(x_test)  # shape of scores: (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the estimator first")

        if x_test.ndim != 2:
            raise ValueError("invalid shape of 'x_test' - should be 2d")

        scores = compute_kde(
            self.x_train,
            self.weights_train,
            x_test,
            self.bandwidth,
            self.kernel_name,
        )
        return scores

    # TODO: https://stats.stackexchange.com/questions/43674/simple-sampling-method-for-a-kernel-density-estimator  # noqa
    def sample(self):
        raise NotImplementedError
