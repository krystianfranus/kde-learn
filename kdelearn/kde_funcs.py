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
from .kde import KDE


class KDEClassifier:
    """Bayes' classifier based on kernel density estimation.

    Probability that :math:`x` belongs to class :math:`c`:

    .. math::
        P(C=c|X=x) \\propto \\pi_c \\hat{f}_c(X=x)

    To predict class label for :math:`x` we need to take class :math:`c` with the
    highest probability:

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
    >>> classifier = KDEClassifier().fit(x_train, labels_train)

    References
    ----------
    - Silverman, B. W. Density Estimation for Statistics and Data Analysis.
      Chapman and Hall, 1986.
    """

    def __init__(self, kernel_name: str = "gaussian"):
        if kernel_name not in kernel_properties:
            available_kernels = list(kernel_properties.keys())
            raise ValueError(f"invalid kernel_name - try one of {available_kernels}")
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
        """Fit the classifier.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n)
            Data points as an array containing data with float type.
        labels_train : ndarray of shape (m_train,)
            Labels of data points as an array containing data with int type.
        weights_train : ndarray of shape (m_train,), default=None
            Weights for data points. If None, all points are equally weighted.
        share_bandwidth : bool, default=False
            Determines whether all classes should have common bandwidth.
            If False, estimator of each class gets its own bandwidth.
        bandwidth_method : {'normal_reference', 'direct_plugin', 'ste_plugin', \
                'ml_cv'}, default='normal_reference'
            Name of bandwidth selection method.
        prior_prob : ndarray of shape (n_classes,), default=None
            Prior probabilities of each class. If None, all classes are equally
            probable.

        Returns
        -------
        self : object
            Fitted self instance of KDEClassifier.

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
        >>> classifier = KDEClassifier().fit(x_train, labels_train, weights_train, prior_prob=prior_prob)  # noqa
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of x_train - should be 2d")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]

        if labels_train.ndim != 1:
            raise ValueError("invalid shape of labels_train - should be 1d")
        if not np.issubdtype(labels_train.dtype, np.integer):
            raise ValueError("invalid dtype of labels_train - should be of int type")
        self.labels_train = labels_train

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            if weights_train.ndim != 1:
                raise ValueError("invalid shape of weights_train - should be 1d")
            if weights_train.shape[0] != x_train.shape[0]:
                raise ValueError("invalid size of weights_train")
            if not (weights_train > 0).all():
                raise ValueError("weights_train must be positive")
            self.weights_train = weights_train / weights_train.sum()

        self.bandwidth = None
        self.bandwidth_method = bandwidth_method

        if share_bandwidth:
            if self.bandwidth_method == "normal_reference":
                self.bandwidth = normal_reference(self.x_train, self.kernel_name)
            elif self.bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                self.bandwidth = direct_plugin(self.x_train, self.kernel_name, stage)
            elif self.bandwidth_method == "ste_plugin":
                self.bandwidth = ste_plugin(self.x_train, self.kernel_name)
            elif self.bandwidth_method == "ml_cv":
                self.bandwidth = ml_cv(
                    self.x_train, self.kernel_name, self.weights_train
                )
            else:
                raise ValueError("invalid bandwidth method")

        self.ulabels = np.unique(labels_train)  # Sorted unique labels
        self.n_classes = self.ulabels.shape[0]
        if prior_prob is None:
            self.prior = self._compute_prior()
        else:
            if prior_prob.ndim != 1:
                raise ValueError("invalid shape of prior_prob - should be 1d")
            if prior_prob.shape[0] != self.n_classes:
                raise ValueError(
                    f"invalid prior_prob - should contain {self.n_classes} values"
                )
            self.prior = prior_prob / prior_prob.sum()

        self.kwargs = kwargs

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        """Predict labels.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Grid data points as an array containing data with float type.

        Returns
        -------
        labels_pred : ndarray of shape (m_test,)
            Predicted labels as a 1D array containing data with int type.

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

        if x_test.ndim != 2:
            raise ValueError("invalid shape of x_test - should be 2d")

        labels_pred, _ = self._classify(x_test)
        return labels_pred

    def pdfs(self, x_test: ndarray) -> ndarray:
        """Compute pdf of each class.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Grid data points as an array containing data with float type.

        Returns
        -------
        scores : ndarray of shape (m_test, n_classes)
            Predicted scores as an array containing data with float type.

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

        if x_test.ndim != 2:
            raise ValueError("invalid shape of x_test - should be 2d")

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
    >>> x_train = np.random.normal(0, 1, size=(100, 1))
    >>> # Fit the outliers detector
    >>> outliers_detector = KDEOutliersDetector("gaussian").fit(x_train)
    """

    def __init__(self, kernel_name: str = "gaussian"):
        if kernel_name not in kernel_properties:
            available_kernels = list(kernel_properties.keys())
            raise ValueError(f"invalid kernel_name - try one of {available_kernels}")
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
        x_train : ndarray of shape (m_train, n)
            Data points as an array containing data with float type.
        weights_train : ndarray of shape (m_train,), default=None
            Weights for data points. If None is passed, all points are equally weighted.
        bandwidth : ndarray of shape (n,), optional
            Smoothing parameter.
        bandwidth_method : {'normal_reference', 'direct_plugin'}, \
                default='normal_reference'
            Name of bandwidth selection method used to compute it when bandwidth
            argument is not passed explicitly.
        r : float
            Threshold.

        Returns
        -------
        self : object
            Fitted self instance of KDEOutliersDetector.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(100, 1))
        >>> weights_train = np.random.uniform(0, 1, size=(100,))
        >>> # Fit the outliers detector
        >>> outliers_detector = KDEOutliersDetector().fit(x_train, weights_train, r=0.1)
        """
        if r < 0 or r > 1:
            raise ValueError("invalid value of r - should be in range [0, 1]")

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
        x_test : ndarray of shape (m_test, n)
            Grid data points as a 2D array containing data with float type.

        Returns
        -------
        labels_pred : ndarray of shape (m_test,)
            Predicted labels (0 - inlier, 1 - outlier) as an array containing data
            with int type.

        Examples
        --------
        >>> # Prepare data
        >>> x_train = np.random.normal(0, 1, size=(100, 1))
        >>> x_test = np.random.uniform(-3, 3, size=(10, 1))
        >>> # Fit the outliers detector
        >>> outliers_detector = KDEOutliersDetector().fit(x_train, r=0.1)
        >>> # Predict the labels
        >>> labels_pred = outliers_detector.predict(x_test)  # labels_pred shape (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the outliers detector first")

        if len(x_test.shape) != 2:
            raise ValueError("invalid shape of array - should be two-dimensional")

        scores = self.kde.pdf(x_test)
        labels_pred = np.where(scores <= self.threshold, 1, 0)
        return labels_pred
