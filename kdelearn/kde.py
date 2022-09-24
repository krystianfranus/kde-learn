from typing import Optional

import numpy as np
from numpy import ndarray

from .bandwidth_selection import (
    direct_plugin,
    kernel_properties,
    ml_cv,
    normal_reference,
    ste_plugin,
)
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
    >>> m_train = 100
    >>> n = 1
    >>> x_train = np.random.normal(0, 1, size=(m_train, n))
    >>> # Fit
    >>> kde = KDE("gaussian").fit(x_train)

    References
    ----------
    [1] Silverman, B. W. Density Estimation for Statistics and Data Analysis.
    Chapman and Hall, 1986.

    [2] Wand, M. P., Jones M.C. Kernel Smoothing. Chapman and Hall, 1995.
    """

    def __init__(self, kernel_name: str = "gaussian"):
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
    ):
        """Fit the estimator.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n)
            Data points as an array containing data with float type.
        weights_train : ndarray of shape (m_train,), optional
            Weights of data points. If None, all points are equally weighted.
        bandwidth : ndarray of shape (n,), optional
            Smoothing parameter.
        bandwidth_method : {'normal_reference', 'direct_plugin', 'ste_plugin', \
                'ml_cv'}, default='normal_reference'
            Name of bandwidth selection method used to compute smoothing parameter
            when `bandwidth` is not given explicitly.

        Returns
        -------
        self : object
            Fitted self instance of KDE.

        Examples
        --------
        >>> # Prepare data
        >>> m_train = 100
        >>> n = 1
        >>> x_train = np.random.normal(0, 1, size=(m_train, n))
        >>> weights_train = np.random.randint(1, 10, size=(m_train,))
        >>> bandwidth = np.random.uniform(0, 1, size=(n,))
        >>> # Fit the estimator
        >>> kde = KDE().fit(x_train, weights_train, bandwidth)
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of 'x_train' - should be 2d")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]
        self.n = self.x_train.shape[1]

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            if weights_train.ndim != 1:
                raise ValueError("invalid shape of 'weights_train' - should be 1d")
            if weights_train.shape[0] != x_train.shape[0]:
                raise ValueError("invalid size of 'weights_train'")
            if not (weights_train > 0).all():
                raise ValueError("'weights_train' should be positive")
            self.weights_train = weights_train / weights_train.sum()

        if bandwidth is None:
            if bandwidth_method == "normal_reference":
                self.bandwidth = normal_reference(self.x_train, self.kernel_name)
            elif bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                self.bandwidth = direct_plugin(self.x_train, self.kernel_name, stage)
            elif bandwidth_method == "ste_plugin":
                self.bandwidth = ste_plugin(self.x_train, self.kernel_name)
            elif bandwidth_method == "ml_cv":
                self.bandwidth = ml_cv(
                    self.x_train, self.kernel_name, self.weights_train
                )
            else:
                raise ValueError("invalid 'bandwidth_method'")
        else:
            if bandwidth.ndim != 1:
                raise ValueError("invalid shape of 'bandwidth' - should be 1d")
            if bandwidth.shape[0] != self.n:
                raise ValueError(
                    f"invalid size of 'bandwidth' - should contain {self.n} values"
                )
            if not (bandwidth > 0).all():
                raise ValueError("'bandwidth' should be positive")
            self.bandwidth = bandwidth

        self.fitted = True
        return self

    def pdf(self, x_test: ndarray) -> ndarray:
        """Compute probability density function.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Grid data points as an array containing data with float type.

        Returns
        -------
        scores : ndarray of shape (m_test,)
            Values of kernel density estimator.

        Examples
        --------
        >>> # Prepare data
        >>> m_train = 100
        >>> n = 1
        >>> x_train = np.random.normal(0, 1, (m_train, n))
        >>> x_test = np.random.uniform(-3, 3, (10, n))
        >>> # Fit the estimator.
        >>> kde = KDE().fit(x_train)
        >>> # Compute pdf
        >>> scores = kde.pdf(x_test)  # scores shape (10,)
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
