from typing import Optional

import numpy as np
from numpy import ndarray

from .kde import Kde
from .utils import estimate_bandwidth


def kde_classifier(
    x_train: ndarray,
    labels_train: ndarray,
    x_test: ndarray,
    weights_train: Optional[ndarray] = None,
    kernel_name: str = "gaussian",
    shared_bandwidth: bool = True,
    prior: Optional[ndarray] = None,
) -> ndarray:
    """
    Bayes' classifier based on kernel density estimation.

    .. math::
        P(C=c|X=x) \\propto \\alpha_c \\hat{f}_c(X=x)
    .. math::
        \\underset{c}{\\mathrm{argmax}} \\ P(C=c|X=x)

    Parameters
    ----------
    x_train : `ndarray`
        Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
    labels_train : `ndarray`
        Data points as a 1D array containing data with `int` type. Must have shape (m_train,).
    x_test : `ndarray`
        Grid data points as a 2D array containing data with `float` type. Must have shape (m_test, n).
    weights_train : `ndarray`, default=None
        Weights for data points. Must have shape (m_train,). If None is passed, all points get the same weights.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    shared_bandwidth : bool, default=True
        Determines whether all classes should have common bandwidth. If False, each class gets own bandwidth.
    prior : `ndarray`, default=None
        Prior probability.

    Returns
    -------
    labels_pred : `ndarray`
        Predicted labels.
    """
    ulabels = np.unique(labels_train)  # sorted unique labels
    if prior is None:
        prior = np.zeros(ulabels.shape)
        for idx, label in enumerate(ulabels):
            mask = labels_train == label
            prior[idx] = labels_train[mask].shape[0] / x_train.shape[0]

    bandwidth = estimate_bandwidth(x_train) if shared_bandwidth else None

    scores = np.zeros((x_test.shape[0], ulabels.shape[0]))
    for idx, label in enumerate(ulabels):
        mask = labels_train == label
        weights = None if weights_train is None else weights_train[mask]
        kde = Kde(kernel_name).fit(x_train[mask], weights, bandwidth)
        scores[:, idx] = kde.pdf(x_test)

    labels_pred = ulabels[np.argmax(scores * prior, axis=1)]
    return labels_pred


def kde_outliers(
    x_train: ndarray,
    x_test: ndarray,
    weights_train: Optional[ndarray] = None,
    r: float = 0.1,
    kernel_name: str = "gaussian",
    bandwidth: ndarray = None,
) -> ndarray:
    """
    Outliers detection based on kernel density estimation.

    Parameters
    ----------
    x_train : `ndarray`
        Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
    x_test : `ndarray`
        Grid data points as a 2D array containing data with `float` type. Must have shape (m_test, n).
    weights_train : `ndarray`, default=None
        Weights for data points. Must have shape (m_train,). If None is passed, all points get the same weights.
    r : float, default=0.1
        Threshold.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    bandwidth : `ndarray`, optional
        Smoothing parameter. Must have shape (n,).

    Returns
    -------
    outliers : `ndarray`
        Indices of detected outliers.
    """
    m_train = x_train.shape[0]
    scores_train = np.empty(m_train)
    for i in range(m_train):
        tmp_x = np.delete(x_train, i, axis=0)
        tmp_weights = np.delete(weights_train, i) if weights_train is not None else None
        kde = Kde(kernel_name).fit(tmp_x, tmp_weights, bandwidth)
        scores_train[i] = kde.score_samples(x_train[[i]])
    q = np.quantile(scores_train, r)

    kde = Kde(kernel_name).fit(x_train, weights_train, bandwidth)
    scores_test = kde.score_samples(x_test)
    outliers = np.where(scores_test <= q)[0]
    return outliers
