from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .bandwidth_selection import direct_plugin, normal_reference, ste_plugin
from .kde import KDE


class KDEClassifier:
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
    >>> # Prepare data for two classes
    >>> x_train1 = np.random.normal(0, 1, size=(100 // 2, 1))
    >>> labels_train1 = np.full(100 // 2, 1)
    >>> x_train2 = np.random.normal(3, 1, size=(100 // 2, 1))
    >>> labels_train2 = np.full(100 // 2, 2)
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> labels_train = np.concatenate((labels_train1, labels_train2))
    >>> # Fit the classifier
    >>> classifier = KDEClassifier("gaussian").fit(x_train, labels_train)
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
        bandwidth_method: str = "normal_reference",
        prior_prob: Optional[ndarray] = None,
        **kwargs,
    ):
        """Fit the classifier to the data.

        Parameters
        ----------
        x_train : `ndarray`
            Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
        labels_train : `ndarray`
            Labels of data points as a 1D array containing data with `int` type. Must have shape (m_train,).
        weights_train : `ndarray`, default=None
            Weights for data points. Must have shape (m_train,). If None, all points are equally weighted.
        share_bandwidth : bool, default=False
            Determines whether all classes should have common bandwidth. If False, estimator of each class gets its own bandwidth.
        bandwidth_method : {'normal_reference', 'direct_plugin', 'ste_plugin'}, default='normal_reference'
            Name of bandwidth selection method.
        prior_prob : `ndarray`, default=None
            Prior probabilities of each class. Must have shape (n_classes,). If None, all classes are equally probable.

        Returns
        -------
        self : `KDEClassifier`
            Fitted self instance of `KDEClassifier`.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> x_train1 = np.random.normal(0, 1, size=(100 // 2, 1))
        >>> labels_train1 = np.full((100 // 2,), 1)
        >>> x_train2 = np.random.normal(3, 1, size=(100 // 2, 1))
        >>> labels_train2 = np.full((100 // 2,), 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> weights_train = np.random.uniform(0, 1, size=(100,))
        >>> # Fit the classifier
        >>> prior_prob = np.array([0.3, 0.7])
        >>> classifier = KDEClassifier().fit(x_train, labels_train, weights_train, prior_prob=prior_prob)
        """
        if len(x_train.shape) != 2:
            raise RuntimeError("x_train must be 2d ndarray")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]

        if len(labels_train.shape) != 1:
            raise RuntimeError("labels_train must be 1d ndarray")
        self.labels_train = labels_train

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            if len(weights_train.shape) != 1:
                raise RuntimeError("weights_train must be 1d ndarray")
            if not (weights_train > 0).all():
                raise ValueError("weights_train must be positive")
            self.weights_train = weights_train / weights_train.sum()

        self.ulabels = np.unique(labels_train)  # Sorted unique labels
        self.n_classes = self.ulabels.shape[0]

        if prior_prob is None:
            self.prior = self._compute_prior()
        else:
            if len(prior_prob.shape) != 1:
                raise RuntimeError("prior_prob must be 1d ndarray")
            if prior_prob.shape[0] != self.n_classes:
                raise RuntimeError(f"prior_prob must contain {self.n_classes} values")
            self.prior = prior_prob / prior_prob.sum()

        self.bandwidth_method = bandwidth_method
        if share_bandwidth:
            if self.bandwidth_method == "normal_reference":
                self.bandwidth = normal_reference(self.x_train, self.kernel_name)
            elif self.bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                self.bandwidth = direct_plugin(self.x_train, self.kernel_name, stage)
            elif self.bandwidth_method == "ste_plugin":
                self.bandwidth = ste_plugin(self.x_train, self.kernel_name)
            else:
                raise ValueError("invalid bandwidth method")
        else:
            self.bandwidth = None

        self.kwargs = kwargs

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
        >>> # Prepare data for two classes
        >>> x_train1 = np.random.normal(0, 1, size=(100 // 2, 1))
        >>> labels_train1 = np.full(100 // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(100 // 2, 1))
        >>> labels_train2 = np.full(100 // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> # Fit the classifier
        >>> x_test = np.random.uniform(-1, 4, size=(10, 1))
        >>> classifier = KDEClassifier().fit(x_train, labels_train)
        >>> # Predict labels
        >>> labels_pred = classifier.predict(x_test)  # labels_pred shape (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the model first")
        labels_pred, _ = self._classify(x_test)
        return labels_pred

    def pdfs(self, x_test: ndarray):
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
        >>> # Prepare data for two classes
        >>> x_train1 = np.random.normal(0, 1, size=(100 // 2, 1))
        >>> labels_train1 = np.full(100 // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(100 // 2, 1))
        >>> labels_train2 = np.full(100 // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> # Fit the classifier
        >>> x_test = np.random.uniform(-1, 4, size=(10, 1))
        >>> classifier = KDEClassifier().fit(x_train, labels_train)
        >>> # Compute pdf of each class
        >>> scores = classifier.pdfs(x_test)  # scores shape (10, 2)
        """
        if not self.fitted:
            raise RuntimeError("fit the classifier first")
        _, scores = self._classify(x_test)
        return scores

    def _compute_prior(self) -> ndarray:
        prior = np.empty(self.ulabels.shape)
        for idx, label in enumerate(self.ulabels):
            mask = self.labels_train == label
            prior[idx] = self.labels_train[mask].shape[0] / self.m_train
        return prior

    def _classify(self, x_test: ndarray) -> Tuple[ndarray, ndarray]:
        scores = np.empty((x_test.shape[0], self.n_classes))
        for idx, label in enumerate(self.ulabels):
            mask = self.labels_train == label
            weights = self.weights_train
            if self.weights_train is not None:
                weights = self.weights_train[mask]
            kde = KDE(self.kernel_name).fit(
                self.x_train[mask],
                weights,
                self.bandwidth,
                self.bandwidth_method,
                **self.kwargs,
            )
            scores[:, idx] = kde.pdf(x_test)

        labels_pred = self.ulabels[np.argmax(self.prior * scores, axis=1)]
        return labels_pred, scores


class KDEOutliersDetector:
    """Outliers Detector.

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data
    >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
    >>> # Fit the outliers detector
    >>> outliers_detector = KDEOutliersDetector("gaussian").fit(x_train)
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
        r: float = 0.1,
        **kwargs,
    ):
        """Fit the outliers detector.

        Parameters
        ----------
        x_train : `ndarray`
            Data points as a 2D array containing data with `float` type. Must have shape (m_train, n).
        weights_train : `ndarray`, default=None
            Weights for data points. Must have shape (m_train,). If None is passed, all points are equally weighted.
        bandwidth : `ndarray`, optional
            Smoothing parameter. Must have shape (n,).
        bandwidth_method : {'normal_reference', 'direct_plugin'}, default='normal_reference'
            Name of bandwidth selection method used to compute it when bandwidth argument is not passed explicitly.
        r : `float`
            Threshold.

        Returns
        -------
        self : `KDEOutliersDetector`
            Fitted self instance of `KDEOutliersDetector`.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
        >>> weights_train = np.random.uniform(0, 1, size=(10_000,))
        >>> # Fit the outliers detector
        >>> outliers_detector = KDEOutliersDetector().fit(x_train, weights_train, r=0.1)
        """
        if r < 0:
            raise ValueError("r must be positive")

        self.kde = KDE(self.kernel_name).fit(
            x_train, weights_train, bandwidth, bandwidth_method, **kwargs
        )
        scores = self.kde.pdf(x_train)
        self.threshold = np.quantile(scores, r)

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        """Predict the labels.

        Parameters
        ----------
        x_test : `ndarray`
            Grid data points as a 2D array containing data with `float` type. Must have shape (m_test, n).

        Returns
        -------
        labels_pred : `ndarray`
            Predicted labels (0 - inlier, 1 - outlier) as a 1D array containing data with `int` type. Shape (m_test,).

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(10_000, 1))
        >>> x_test = np.random.uniform(-3, 3, size=(1000, 1))
        >>> # Fit the outliers detector
        >>> outliers_detector = KDEOutliersDetector().fit(x_train, r=0.1)
        >>> # Predict the labels
        >>> labels_pred = outliers_detector.predict(x_test)  # labels_pred shape (1000,)
        """
        if not self.fitted:
            raise RuntimeError("fit the outliers detector first")

        if len(x_test.shape) != 2:
            raise ValueError("x_test must be 2d ndarray")

        scores = self.kde.pdf(x_test)
        labels_pred = np.where(scores <= self.threshold, 1, 0)
        return labels_pred
