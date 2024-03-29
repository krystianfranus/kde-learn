from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .bandwidth_selection import normal_reference
from .kde import KDE


def accuracy_loo(
    x_train: ndarray,
    labels_train: ndarray,
    model,
    **kwargs,
) -> float:
    """Leave-one-out accuracy - ratio of correctly classified data points based on
    leave-one-out approach.

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points as an array containing data with float type.
    labels_train : ndarray of shape (m_train,)
        Labels of data points as an array containing data with int type.
    model
        Classifier with defined `fit` and `predict` methods.

    Returns
    -------
    accuracy : float
        Leave-one-out accuracy.

    Examples
    --------
    >>> # Prepare data for two classes
    >>> x_train1 = np.random.normal(0, 1, size=(100 // 2, 1))
    >>> labels_train1 = np.full(100 // 2, 1)
    >>> x_train2 = np.random.normal(3, 1, size=(100 // 2, 1))
    >>> labels_train2 = np.full(100 // 2, 2)
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> labels_train = np.concatenate((labels_train1, labels_train2))
    >>> # Classify and compute accuracy
    >>> model = KDEClassification()
    >>> accuracy = accuracy_loo(x_train, labels_train, model)
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of 'x_train' - should be 2d")
    m_train = x_train.shape[0]

    if labels_train.ndim != 1:
        raise ValueError("invalid shape of 'labels_train' - should be 1d")
    if not np.issubdtype(labels_train.dtype, np.integer):
        raise ValueError("invalid dtype of 'labels_train' - should be of int type")

    if not hasattr(model, "fit"):
        raise AttributeError(f"'{model}' object has no attribute 'fit'")
    if not hasattr(model, "predict"):
        raise AttributeError(f"'{model}' object has no attribute 'predict'")

    labels_pred = np.empty((m_train,), dtype=np.int32)
    for i in range(m_train):
        mask = np.delete(np.arange(m_train), i)
        classifier = model.fit(
            x_train[mask],
            labels_train[mask],
            **kwargs,
        )
        labels_pred[i] = classifier.predict(x_train[i : i + 1])

    accuracy = np.sum(labels_train == labels_pred) / m_train
    return accuracy


def pi_kf(
    x_train: ndarray,
    x_test: ndarray,
    labels_test: ndarray,
    weights_train: Optional[ndarray] = None,
) -> float:
    """Performance index for outliers detection.

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n_x)
        Data points as an array containing data with float type.
    x_test : ndarray of shape (m_test, n_x)
        Data points as an array containing data with float type.
    labels_test : ndarray of shape (m_test,)
        Labels (0 - inlier, 1 - outlier) of data points as an array containing data
        with int type.
    weights_train : ndarray of shape (m_train,), optional
        Weights of data points. If None, all points are equally weighted.

    Returns
    -------
    pi : float
        Performance index.

    Examples
    --------
    >>> x_train = np.array([[-0.1], [0.0], [0.1], [1.1]])
    >>> labels_train = np.array([0, 0, 0, 1])
    >>> pi = pi_kf(x_train, labels_train)
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of 'x_train' - should be 2d")

    if x_test.ndim != 2:
        raise ValueError("invalid shape of 'x_test' - should be 2d")

    if labels_test.ndim != 1:
        raise ValueError("invalid shape of 'labels_test' - should be 1d")
    if not np.issubdtype(labels_test.dtype, np.integer):
        raise ValueError("invalid dtype of 'labels_test' - should be of int type")
    if not np.all(np.isin(labels_test, [0, 1])):
        raise ValueError("invalid values in 'labels_test' - should contain 0 or 1")

    inliers = labels_test == 0
    outliers = labels_test == 1
    n_outliers = (outliers == 1).sum()

    kde = KDE().fit(x_train, weights_train, bandwidth_method="direct_plugin")
    scores = kde.pdf(x_test)
    scores_out = scores[outliers]
    idx = np.argsort(scores[inliers])[:n_outliers]
    scores_in = scores[inliers][idx]

    pi = np.sum(scores_out) / np.sum(scores_in)
    return pi


def density_silhouette(
    x_test: ndarray,
    labels_test: ndarray,
    weights_test: Optional[ndarray] = None,
    weights2: Optional[ndarray] = None,
    kernel_name: str = "gaussian",
    share_bandwidth: bool = False,
) -> Tuple[ndarray, float]:
    """Density based silhouette.

    Parameters
    ----------
    x_test : ndarray of shape (m_test, n)
        Data points as an array containing data with float type.
    labels_test : ndarray of shape (m_test,)
        Labels of data points as an array containing data with int type.
    weights_test : ndarray of shape (m_test,), default=None
        Weights of data points. If None, all points are equally weighted.
    weights2 : ndarray of shape (m_test,), default=None
        Weights of data points. If None, all points are equally weighted.
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.
    share_bandwidth : bool, default=False
        Determines whether all clusters should have common bandwidth.
        If False, estimator of each cluster gets its own bandwidth.

    Returns
    -------
    dbs : ndarray of shape (m_train,)
        Density based silhouette scores of all data points.
    dbs_mean : float
        Mean density based silhouette score.

    Examples
    --------
    >>> x_train = np.array([[-0.1], [0.0], [0.1], [2.9], [3.0], [3.1]])
    >>> labels_train = np.array([0, 0, 0, 1, 1 ,1])
    >>> dbs, dbs_mean = density_silhouette(x_train, labels_train)

    References
    ----------
    [1] Menardi, G. Density-based Silhouette diagnostics for clustering methods.
    Springer, 2010.
    """
    if x_test.ndim != 2:
        raise ValueError("invalid shape of 'x_train' - should be 2d")

    if labels_test.ndim != 1:
        raise ValueError("invalid shape of 'labels_test' - should be 1d")
    if not np.issubdtype(labels_test.dtype, np.integer):
        raise ValueError("invalid dtype of 'labels_test' - should be of int type")

    m_test, n = x_test.shape
    # Sorted unique labels
    ulabels, cluster_sizes = np.unique(labels_test, return_counts=True)
    n_clusters = ulabels.shape[0]

    if ulabels[0] != 0:
        raise ValueError(
            "invalid values in 'labels_train' - labels should be enumerated from 0"
        )

    if weights_test is None:
        weights_test = np.full(m_test, 1 / m_test)
    else:
        if weights_test.ndim != 1:
            raise ValueError("invalid shape of 'weights_test' - should be 1d")
        if weights_test.shape[0] != x_test.shape[0]:
            raise ValueError("invalid size of 'weights_test'")
        if not (weights_test >= 0).all():
            raise ValueError("'weights_test' must be non negative")
        weights_test = weights_test / weights_test.sum()

    # Prepare bandwidths for each cluster
    if share_bandwidth:
        bandwidth = normal_reference(x_test, None, kernel_name)
        cluster_bandwidths = np.full((n_clusters, n), bandwidth)
        valid_bandwidths = np.full(n_clusters, True)
    else:
        cluster_bandwidths = np.empty((n_clusters, n))
        valid_bandwidths = np.full(n_clusters, False)
        for idx, label in enumerate(ulabels):
            if cluster_sizes[idx] != 1:
                x_train_tmp = x_test[labels_test == label]
                cluster_bandwidths[idx] = normal_reference(
                    x_train_tmp, None, kernel_name
                )
                valid_bandwidths[idx] = True
    valid_bandwidths = valid_bandwidths[:, None]
    bandwidth_mean = np.mean(cluster_bandwidths, axis=0, where=valid_bandwidths)

    # Compute dbs
    theta = np.empty((n_clusters, m_test))
    for idx, label in enumerate(ulabels):
        cluster_size = cluster_sizes[idx]
        bandwidth = cluster_bandwidths[idx] if cluster_size != 1 else bandwidth_mean
        kde = KDE(kernel_name).fit(
            x_test[labels_test == label],
            weights_train=weights_test[labels_test == label],
            bandwidth=bandwidth,
        )
        scores = kde.pdf(x_test)
        theta[idx, :] = cluster_size / m_test * scores
    theta = theta / theta.sum(axis=0)

    arange = np.arange(m_test)
    # Posterior probability that x_i belongs to its own cluster
    theta_m0 = theta[labels_test, arange]
    theta[labels_test, arange] = 0
    # Posterior probability that x_i belongs to the nearest cluster
    theta_m1 = np.max(theta, axis=0)

    # Smallest positive float number - preventing from computing log(0)
    e = np.nextafter(0, 1)
    dbs = (np.log(theta_m0 + e) - np.log(theta_m1 + e)) / np.max(
        np.abs((np.log(theta_m0 + e) - np.log(theta_m1 + e)))
    )
    dbs_mean = np.average(dbs, weights=weights2)

    return dbs, dbs_mean
