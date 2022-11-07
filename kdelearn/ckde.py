from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .bandwidth_selection import (
    direct_plugin,
    kernel_properties,
    ml_cv,
    normal_reference,
    ste_plugin,
)
from .cutils import compute_d, compute_kde


class CKDE:
    """Conditional kernel density estimator with product kernel:

    TODO: <MATH FORMULA and READ MORE and REFERENCES>

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data
    >>> m_train = 100
    >>> n_x, n_y = 1, 1
    >>> x_train = np.random.normal(0, 1, size=(m_train, n_x))
    >>> y_train = np.random.normal(0, 1, size=(m_train, n_y))
    >>> y_star = np.array([0.0] * n_y)
    >>> # Fit
    >>> ckde = CKDE("gaussian").fit(x_train, y_train, y_star)
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
        y_train: ndarray,
        y_star: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth_x: Optional[ndarray] = None,
        bandwidth_y: Optional[ndarray] = None,
        bandwidth_method: str = "normal_reference",
        **kwargs,
    ):
        """Fit the estimator.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n_x)
            Data points (describing variables) as an array containing data with float
            type.
        y_train : ndarray of shape (m_train, n_y)
            Data points (conditioning variables) as an array containing data with float
            type.
        y_star : ndarray of shape (n_y,)
            Conditioned value.
        weights_train : ndarray of shape (m_train,), optional
            Weights of data points. If None, all points are equally weighted.
        bandwidth_x : ndarray of shape (n_x,), optional
            Smoothing parameter of describing variables.
        bandwidth_y : ndarray of shape (n_y,), optional
            Smoothing parameter of conditioning variables.
        bandwidth_method : {'normal_reference', 'direct_plugin', 'ste_plugin', \
                'ml_cv'}, default='normal_reference'
            Name of bandwidth selection method used to compute smoothing parameter
            when `bandwidth` is not given explicitly.

        Returns
        -------
        self : object
            Fitted self instance of CKDE.

        Examples
        --------
        >>> # Prepare data
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> x_train = np.random.normal(0, 1, size=(m_train, n_x))
        >>> y_train = np.random.normal(0, 1, size=(m_train, n_y))
        >>> weights_train = np.random.randint(1, 10, size=(m_train,))
        >>> y_star = np.array([0.0] * n_y)
        >>> bandwidth_x = np.array([0.5] * n_x)
        >>> bandwidth_y = np.array([0.5] * n_y)
        >>> # Fit the estimator
        >>> params = (x_train, y_train, y_star, weights_train, bandwidth_x, bandwidth_y)
        >>> ckde = CKDE().fit(*params)
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of 'x_train' - should be 2d")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]
        self.n_x = self.x_train.shape[1]

        if y_train.ndim != 2:
            raise ValueError("invalid shape of 'y_train' - should be 2d")
        if y_train.shape[0] != x_train.shape[0]:
            raise ValueError("invalid size of 'y_train'")
        self.y_train = y_train
        self.n_y = self.y_train.shape[1]

        if y_star.ndim != 1:
            raise ValueError("invalid shape of 'y_star' - should be 1d")
        if y_star.shape[0] != self.n_y:
            raise ValueError(
                f"invalid size of 'y_star'- should contain {self.n_y} values"
            )
        self.y_star = y_star

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

        if bandwidth_x is None or bandwidth_y is None:
            z_train = np.concatenate((self.x_train, self.y_train), axis=1)
            if bandwidth_method == "normal_reference":
                bandwidth = normal_reference(
                    z_train, self.weights_train, self.kernel_name
                )
            elif bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                bandwidth = direct_plugin(
                    z_train, self.weights_train, self.kernel_name, stage
                )
            elif bandwidth_method == "ste_plugin":
                bandwidth = ste_plugin(z_train, self.weights_train, self.kernel_name)
            elif bandwidth_method == "ml_cv":
                bandwidth = ml_cv(z_train, self.kernel_name, self.weights_train)
            else:
                raise ValueError("invalid 'bandwidth_method'")
            self.bandwidth_x = bandwidth[: self.n_x]
            self.bandwidth_y = bandwidth[self.n_x :]
        else:
            if bandwidth_x.ndim != 1:
                raise ValueError("invalid shape of 'bandwidth_x' - should be 1d")
            if bandwidth_y.ndim != 1:
                raise ValueError("invalid shape of 'bandwidth_y' - should be 1d")
            if bandwidth_x.shape[0] != self.n_x:
                raise ValueError(
                    f"invalid size of 'bandwidth_x' - should contain {self.n_x} values"
                )
            if bandwidth_y.shape[0] != self.n_y:
                raise ValueError(
                    f"invalid size of 'bandwidth_y' - should contain {self.n_y} values"
                )
            if not (bandwidth_x > 0).all():
                raise ValueError("'bandwidth_x' should be positive")
            if not (bandwidth_y > 0).all():
                raise ValueError("'bandwidth_y' should be positive")
            self.bandwidth_x = bandwidth_x
            self.bandwidth_y = bandwidth_y

        self.fitted = True
        return self

    def pdf(
        self,
        x_test: ndarray,
    ) -> Tuple[ndarray, ndarray]:
        """Compute conditional probability density function.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n_x)
            Grid data points (describing variables) as an array containing data with
            float type.

        Returns
        -------
        scores : ndarray of shape (m_test,)
            Values of kernel density estimator.
        cond_weights_train : ndarray of shape (m_train,)
            TODO: complete !!!!!!!!

        Examples
        --------
        >>> # Prepare data
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> m_test = 10
        >>> x_train = np.random.normal(0, 1, (m_train, n_x))
        >>> y_train = np.random.normal(0, 1, (m_train, n_y))
        >>> y_star = np.array([0.0] * n_y)
        >>> x_test = np.random.uniform(-3, 3, (m_test, n_x))
        >>> # Fit the estimator.
        >>> ckde = CKDE().fit(x_train, y_train, y_star)
        >>> # Compute pdf
        >>> scores, d = ckde.pdf(x_test)  # scores shape (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the estimator first")

        if x_test.ndim != 2:
            raise ValueError("invalid shape of 'x_test' - should be 2d")

        cond_weights_train = compute_d(
            self.y_train,
            self.weights_train,
            self.y_star,
            self.bandwidth_y,
            self.kernel_name,
        )
        scores = compute_kde(
            self.x_train,
            cond_weights_train,
            x_test,
            self.bandwidth_x,
            self.kernel_name,
        )
        return scores, cond_weights_train
