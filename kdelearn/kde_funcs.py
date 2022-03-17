from typing import Optional

import numpy as np
from numpy import ndarray

from .kde import Kde
from .utils import scotts_rule


class KdeClassifier:
    """Bayes' classifier based on kernel density estimation.

    Probability that :math:`x` belongs to class :math:`c`:

    .. math::
        P(C=c|X=x) \\propto \\pi_c \\hat{f}_c(X=x)

    To predict class label for :math:`x` we need to take class :math:`c` with the highest probability:

    .. math::
        \\underset{c}{\\mathrm{argmax}} \\quad P(C=c|X=x)

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data points and labels for two classes
    >>> x_train1 = np.random.normal(0, 1, size=(10_000 // 2, 1))
    >>> labels_train1 = np.full(10_000 // 2, 1)
    >>> x_train2 = np.random.normal(3, 1, size=(10_000 // 2, 1))
    >>> labels_train2 = np.full(10_000 // 2, 2)
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> labels_train = np.concatenate((labels_train1, labels_train2))
    >>> # Classifier
    >>> model = KdeClassifier().fit(x_train, labels_train)
    """

    def __init__(self, kernel_name: str = "gaussian"):
        self.kernel_name = kernel_name
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        labels_train: ndarray,
        weights_train: Optional[ndarray] = None,
        share_bandwidth: bool = False,
        prior_prob: Optional[ndarray] = None,
    ):
        """Fit model.

        Parameters
        ----------
        x_train : `ndarray`
            Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
        labels_train : `ndarray`
            Labels of data points as a 1D array containing data with `int` type. Must have shape (m_train,).
        weights_train : `ndarray`, default=None
            Weights for data points. Must have shape (m_train,). If None is passed, all points get the same weights.
        share_bandwidth : bool, default=False
            Determines whether all classes should have common bandwidth. If False, each class gets its own bandwidth.
        prior_prob : `ndarray`, default=None
            Prior probabilities of each class. Must have shape (n_classes,).

        Returns
        -------
        self : `KdeClassifier`
            Fitted self instance of `KdeClassifier`.

        Examples
        --------
        >>> # Prepare data points and labels for two classes
        >>> x_train1 = np.random.normal(0, 1, size=(10_000 // 2, 1))
        >>> labels_train1 = np.full((10_000 // 2,), 1)
        >>> x_train2 = np.random.normal(3, 1, size=(10_000 // 2, 1))
        >>> labels_train2 = np.full((10_000 // 2,), 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> weights_train = np.random.uniform(0, 1, size=(10_000,))
        >>> prior_prob = np.array([0.3, 0.7])
        >>> # Classifier
        >>> model = KdeClassifier().fit(x_train, labels_train, weights_train, share_bandwidth=True, prior_prob=prior_prob)
        """
        if len(x_train.shape) != 2:
            raise RuntimeError("x_train must be 2d ndarray")
        self.x_train = x_train

        if len(labels_train.shape) != 1:
            raise RuntimeError("labels_train must be 1d ndarray")
        self.labels_train = labels_train

        if weights_train is not None:
            if len(weights_train.shape) != 1:
                raise RuntimeError("weights_train must be 1d ndarray")
        self.weights_train = weights_train

        self.ulabels = np.unique(labels_train)  # sorted unique labels
        self.n_classes = self.ulabels.shape[0]

        if prior_prob is None:
            self.prior = self._compute_prior()
        else:
            if len(prior_prob.shape) != 1:
                raise RuntimeError("prior_prob must be 1d ndarray")
            if prior_prob.shape[0] != self.n_classes:
                raise RuntimeError(f"prior_prob must contain {self.n_classes} values")
            self.prior = prior_prob / prior_prob.sum()  # l1 norm

        self.bandwidth = scotts_rule(x_train) if share_bandwidth else None
        self.fitted = True
        return self

    def predict(self, x_test: ndarray):
        """Predict labels.

        Parameters
        ----------
        x_test : `ndarray`
            Grid data points as a 2D array containing data with `float` type. Must have shape (m_test, n).

        Returns
        -------
        labels_pred : `ndarray`
            Predicted labels as a 1D array containing data with `int` type. Shape (m_test,).

        Examples
        --------
        >>> # Prepare data points and labels for two classes
        >>> x_train1 = np.random.normal(0, 1, size=(10_000 // 2, 1))
        >>> labels_train1 = np.full(10_000 // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(10_000 // 2, 1))
        >>> labels_train2 = np.full(10_000 // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> # Classify
        >>> x_test = np.random.uniform(-1, 4, size=(1000, 1))
        >>> model = KdeClassifier().fit(x_train, labels_train)
        >>> labels_pred = model.predict(x_test)  # labels_pred shape (1000,)
        """
        if not self.fitted:
            raise RuntimeError("fit the model first")
        labels_pred, _ = self._classify(x_test)
        return labels_pred

    def score(self, x_test: ndarray):
        """Compute pdf of each class.

        Parameters
        ----------
        x_test : `ndarray`
            Grid data points as a 2D array containing data with `float` type. Must have shape (m_test, n).

        Returns
        -------
        scores : `ndarray`
            Predicted scores as a 2D array containing data with `float` type. Shape (m_test, n_classes).

        Examples
        --------
        >>> # Prepare data points and labels for two classes
        >>> x_train1 = np.random.normal(0, 1, size=(10_000 // 2, 1))
        >>> labels_train1 = np.full(10_000 // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(10_000 // 2, 1))
        >>> labels_train2 = np.full(10_000 // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> # Classify
        >>> x_test = np.random.uniform(-1, 4, size=(1000, 1))
        >>> model = KdeClassifier().fit(x_train, labels_train)
        >>> scores = model.score(x_test)  # scores shape (1000, 2)
        """
        if not self.fitted:
            raise RuntimeError("fit the model first")
        _, scores = self._classify(x_test)
        return scores

    def _compute_prior(self):
        prior = np.empty(self.ulabels.shape)
        for idx, label in enumerate(self.ulabels):
            mask = self.labels_train == label
            prior[idx] = self.labels_train[mask].shape[0] / self.x_train.shape[0]
        return prior

    def _classify(self, x_test: ndarray):
        scores = np.empty((x_test.shape[0], self.n_classes))
        for idx, label in enumerate(self.ulabels):
            mask = self.labels_train == label
            weights = self.weights_train
            if self.weights_train is not None:
                weights = self.weights_train[mask]
            kde = Kde(self.kernel_name).fit(self.x_train[mask], weights, self.bandwidth)
            scores[:, idx] = kde.pdf(x_test)

        labels_pred = self.ulabels[np.argmax(self.prior * scores, axis=1)]
        return labels_pred, scores


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

    Examples
    --------
    >>> # Prepare data points
    >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> # Detect outliers
    >>> x_test = np.random.uniform(-2, 2, size=(1000, 1))
    >>> outliers = kde_outliers(x_train, x_test, r=0.1)

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
        scores_train[i] = kde.pdf(x_train[[i]])
    q = np.quantile(scores_train, r)

    kde = Kde(kernel_name).fit(x_train, weights_train, bandwidth)
    scores_test = kde.pdf(x_test)
    outliers = np.where(scores_test <= q)[0]
    return outliers
