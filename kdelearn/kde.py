from typing import Optional

import numpy as np
from numpy import ndarray

from kdelearn.cutils import compute_kde

from .bandwidth_selection import direct_plugin, ml_cv, normal_reference, ste_plugin


class KDE:
    """Kernel density estimator with product kernel:

    .. math::
        \\hat{f}(x) = \\sum_{i=1}^m w_{i} \\prod_{j=i}^n \\frac{1}{h_j}
        K \\left( \\frac{x_{j} - x_{i, j}}{h_j} \\right)

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> # Fit the estimator
    >>> kde = KDE("gaussian").fit(x_train)

    References
    ----------
    - Silverman, B. W. Density Estimation for Statistics and Data Analysis.
      Chapman and Hall, 1986.
    - Wand, M. P., Jones M.C. Kernel Smoothing. Chapman and Hall, 1995.
    """

    def __init__(self, kernel_name: str = "gaussian"):
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
        x_train : `ndarray`
            Data points as a 2D array containing data with `float` type.
            Must have shape (m_train, n).
        weights_train : `ndarray`, optional
            Weights for data points. Must have shape (m_train,).
            If None, all points are equally weighted.
        bandwidth : `ndarray`, optional
            Smoothing parameter. Must have shape (n,).
        bandwidth_method : {'normal_reference', 'direct_plugin', 'ste_plugin', \
                'ml_cv'}, default='normal_reference'
            Name of bandwidth selection method used to compute it when bandwidth
            argument is not passed explicitly.

        Returns
        -------
        self : `KDE`
            Fitted self instance of `KDE`.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(100, 1))
        >>> weights_train = np.random.randint(1, 10, size=(10,))
        >>> bandwidth = np.random.uniform(0, 1, size=(1,))
        >>> # Fit the estimator
        >>> kde = KDE().fit(x_train, weights_train, bandwidth)
        """
        if len(x_train.shape) != 2:
            raise ValueError("x_train must be 2d ndarray")
        self.x_train = x_train

        if weights_train is None:
            m_train = self.x_train.shape[0]
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            if len(weights_train.shape) != 1:
                raise ValueError("weights_train must be 1d ndarray")
            if not (weights_train > 0).all():
                raise ValueError("weights_train must be positive")
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
                raise ValueError("invalid bandwidth method")
        else:
            if not (bandwidth > 0).all():
                raise ValueError("bandwidth must be positive")
            self.bandwidth = bandwidth

        self.fitted = True
        return self

    def pdf(self, x_test: ndarray) -> ndarray:
        """Compute probability density function.

        Parameters
        ----------
        x_test : `ndarray`
            Grid data points as a 2D array containing data with `float` type.
            Must have shape (m_test, n).

        Returns
        -------
        scores : `ndarray`
            Values of kernel density estimator.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, (100, 1))
        >>> x_test = np.random.uniform(-3, 3, (10, 1))
        >>> # Fit the estimator.
        >>> kde = KDE().fit(x_train)
        >>> # Compute pdf
        >>> scores = kde.pdf(x_test)  # scores shape (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the estimator first")

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
