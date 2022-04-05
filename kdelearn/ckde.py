from typing import Optional

import numpy as np
from numpy import ndarray

from kdelearn.cutils import compute_ckde

from .utils import scotts_rule


class CKDE:
    """Conditional kernel density estimator.

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data
    >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> w_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> # Fit the estimator
    >>> ckde = CKDE("gaussian").fit(x_train, w_train)
    """

    def __init__(self, kernel_name: str = "gaussian"):
        self.kernel_name = kernel_name

    def fit(
        self,
        x_train: ndarray,
        w_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth_x: Optional[ndarray] = None,
        bandwidth_w: Optional[ndarray] = None,
    ):
        """Fit.

        Parameters
        ----------
        x_train : `ndarray`
            Data points (explanatory variables) as a 2D array containing data with `float` type. Must have shape (m_train, n_x).
        w_train : `ndarray`
            Data points (conditional variables) as a 2D array containing data with `float` type. Must have shape (m_train, n_w).
        weights_train : `ndarray`, optional
            Weights for data points. Must have shape (m_train,). If None is passed, all points get the same weights.
        bandwidth_x : `ndarray`, optional
            Smoothing parameter for explanatory variables. Must have shape (n_x,).
        bandwidth_w : `ndarray`, optional
            Smoothing parameter for conditional variables. Must have shape (n_w,).

        Returns
        -------
        self : `CKDE`
            Fitted self instance of `CKDE`.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
        >>> w_train = np.random.normal(0, 1, size=(10_000, 1))
        >>> weights_train = np.random.randint(1, 10, size=(10_000,))
        >>> bandwidth_x = np.random.uniform(0, 1, size=(1,))
        >>> bandwidth_w = np.random.uniform(0, 1, size=(1,))
        >>> # Fit the estimator
        >>> ckde = CKDE().fit(x_train, w_train, weights_train, bandwidth_x, bandwidth_w)
        """
        if len(x_train.shape) != 2 or len(w_train.shape) != 2:
            raise RuntimeError("x_train and w_train must be 2d ndarray")
        self.x_train = x_train
        self.w_train = w_train

        if weights_train is None:
            m_train = self.x_train.shape[0]
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            if len(weights_train.shape) != 1:
                raise RuntimeError("weights_train must be 1d ndarray")
            if not (weights_train > 0).all():
                raise ValueError("weights_train must be positive")
            self.weights_train = weights_train / weights_train.sum()

        if bandwidth_x is None or bandwidth_w is None:
            self.bandwidth_x = scotts_rule(self.x_train, self.kernel_name)
            self.bandwidth_w = scotts_rule(self.w_train, self.kernel_name)
        else:
            self.bandwidth_x = bandwidth_x
            self.bandwidth_w = bandwidth_w

        return self

    def pdf(self, x_test: ndarray, w_test: ndarray) -> ndarray:
        """Compute estimation of conditional probability density function.

        Parameters
        ----------
        x_test : `ndarray`
            Grid data points (explanatory variables) as a 2D array containing data with `float` type. Must have shape (m_test, n).
        w_test : `ndarray`
            Grid data points (conditional variables) as a 2D array containing data with `float` type. Must have shape (m_test, n).

        Returns
        -------
        scores : `ndarray`
            Values of conditional kernel density estimator.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
        >>> w_train = np.random.normal(0, 1, size=(10_000, 1))
        >>> x_test = np.random.uniform(-3, 3, size=(1000, 1))
        >>> w_test = np.full((1000, 1), 0.)
        >>> # Fit the estimator
        >>> ckde = CKDE().fit(x_train, w_train)
        >>> # Compute pdf
        >>> scores = ckde.pdf(x_test, w_test)  # scores shape (1000,)
        """
        scores = compute_ckde(
            self.x_train,
            self.w_train,
            x_test,
            w_test,
            self.weights_train,
            self.bandwidth_x,
            self.bandwidth_w,
            self.kernel_name,
        )
        return scores

    # TODO: https://stats.stackexchange.com/questions/43674/simple-sampling-method-for-a-kernel-density-estimator
    def sample(self):
        raise NotImplementedError
