import warnings
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .bandwidth_selection import direct_plugin, kernel_properties, normal_reference
from .cutils import assign_labels, gradient_ascent, mean_shift
from .kde import KDE


class KDEClassification:
    """Bayes' classifier based on kernel density estimation.

    Probability that :math:`x` belongs to class :math:`c`:

    .. math::
        P(C=c|X=x) \\propto \\pi_c \\hat{f}_c(X=x)

    To predict class label for :math:`x` we need to take class :math:`c` with the
    highest probability:

    .. math::
        \\underset{c}{\\mathrm{argmax}} \\quad P(C=c|X=x)

    Read more :ref:`here <unconditional_classification>`.

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data for two classes
    >>> m_train, n = 100, 1
    >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
    >>> labels_train1 = np.full(m_train // 2, 1)
    >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
    >>> labels_train2 = np.full(m_train // 2, 2)
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> labels_train = np.concatenate((labels_train1, labels_train2))
    >>> # Fit
    >>> classifier = KDEClassification("gaussian").fit(x_train, labels_train)

    References
    ----------
    [1] Silverman, B. W. Density Estimation for Statistics and Data Analysis.
    Chapman and Hall, 1986.
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
        labels_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidths: Optional[ndarray] = None,
        bandwidth_method: str = "direct_plugin",
        share_bandwidth: bool = False,
        prior_prob: Optional[ndarray] = None,
        **kwargs,
    ):
        """Fit the classifier.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n)
            Array containing data points with float type for constructing the
            classifier.
        labels_train : ndarray of shape (m_train,)
            Class labels of `x_train` containing data with int type.
        weights_train : ndarray of shape (m_train,), default=None
            Weights for data points. If None, all data points are equally weighted.
        bandwidths : ndarray of shape (n_classes, n), optional
            Smoothing parameters for scaling the estimators of each class. If None,
            `bandwidth_method` is used to compute the `bandwidth`.
        bandwidth_method : {'normal_reference', 'direct_plugin'}, \
                default='normal_reference'
            Name of bandwidth selection method used to compute `bandwidths` when it
            is not given explicitly.
        share_bandwidth : bool, default=False
            Determines whether all classes should have common bandwidth. If False,
            estimator of each class gets its own bandwidth.
        prior_prob : ndarray of shape (n_classes,), default=None
            Prior probabilities of each class. If None, all classes are equally
            probable.

        Returns
        -------
        self : object
            Fitted self instance of KDEClassification.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> m_train, n = 100, 1
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
        >>> labels_train1 = np.full((m_train // 2,), 1)
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
        >>> labels_train2 = np.full((m_train // 2,), 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> weights_train = np.full((m_train,), 1 / m_train)
        >>> # Fit
        >>> prior_prob = np.array([0.3, 0.7])
        >>> params = (x_train, labels_train, weights_train)
        >>> classifier = KDEClassification().fit(*params, prior_prob=prior_prob)
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of 'x_train' - should be 2d")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]
        n = self.x_train.shape[1]

        if labels_train.ndim != 1:
            raise ValueError("invalid shape of 'labels_train' - should be 1d")
        if not np.issubdtype(labels_train.dtype, np.integer):
            raise ValueError("invalid dtype of 'labels_train' - should be of int type")
        self.labels_train = labels_train

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            if weights_train.ndim != 1:
                raise ValueError("invalid shape of 'weights_train' - should be 1d")
            if weights_train.shape[0] != x_train.shape[0]:
                raise ValueError("invalid size of 'weights_train'")
            if not (weights_train >= 0).all():
                raise ValueError("'weights_train' should be non negative")
            self.weights_train = weights_train / weights_train.sum()

        self.ulabels = np.unique(labels_train)  # Sorted unique labels
        self.n_classes = self.ulabels.shape[0]

        self.bandwidth_method = bandwidth_method
        if bandwidths is None:
            if share_bandwidth:
                if self.bandwidth_method == "normal_reference":
                    bandwidth = normal_reference(
                        self.x_train,
                        self.weights_train,
                        self.kernel_name,
                    )
                elif self.bandwidth_method == "direct_plugin":
                    stage = kwargs["stage"] if "stage" in kwargs else 2
                    bandwidth = direct_plugin(
                        self.x_train,
                        self.weights_train,
                        self.kernel_name,
                        stage,
                    )
                else:
                    raise ValueError("invalid 'bandwidth_method'")
                self.bandwidths = np.full((self.n_classes, n), bandwidth)
            else:
                self.bandwidths = np.full((self.n_classes,), None)
        else:
            if bandwidths.ndim != 2:
                raise ValueError("invalid shape of 'bandwidths' - should be 2d")
            if not (bandwidths > 0).all():
                raise ValueError("'bandwidths' should be positive")
            self.bandwidths = bandwidths

        if prior_prob is None:
            self.prior = self._compute_prior()
        else:
            if prior_prob.ndim != 1:
                raise ValueError("invalid shape of 'prior_prob' - should be 1d")
            if prior_prob.shape[0] != self.n_classes:
                raise ValueError(
                    f"invalid size of 'prior_prob' - should contain {self.n_classes} "
                    "values"
                )
            self.prior = prior_prob / prior_prob.sum()

        self.kwargs = kwargs

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        """Predict class labels.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Data points to classify - array containing data points with float type.

        Returns
        -------
        labels_pred : ndarray of shape (m_test,)
            Predicted class labels containing data with int type.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> m_train, n = 100, 1
        >>> m_test = 10
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
        >>> labels_train1 = np.full(m_train // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
        >>> labels_train2 = np.full(m_train // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> # Fit the classifier
        >>> x_test = np.linspace(-3, 6, m_test).reshape(-1, 1)
        >>> classifier = KDEClassification().fit(x_train, labels_train)
        >>> # Predict labels
        >>> labels_pred = classifier.predict(x_test)  # shape: (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the model first")

        if x_test.ndim != 2:
            raise ValueError("invalid shape of 'x_test' - should be 2d")

        labels_pred, _ = self._classify(x_test)
        return labels_pred

    def pdfs(self, x_test: ndarray) -> ndarray:
        """Compute pdf of each class.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Argument of each class estimator - array containing data points with float
            type.

        Returns
        -------
        scores : ndarray of shape (m_test, n_classes)
            Predicted scores as an array containing data with float type.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> m_train, n = 100, 1
        >>> m_test = 10
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
        >>> labels_train1 = np.full(m_train // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
        >>> labels_train2 = np.full(m_train // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> # Fit the classifier
        >>> x_test = np.linspace(-3, 6, m_test).reshape(-1, 1)
        >>> classifier = KDEClassification().fit(x_train, labels_train)
        >>> # Compute pdf of each class
        >>> scores = classifier.pdfs(x_test)  # shape: (10, 2)
        """
        if not self.fitted:
            raise RuntimeError("fit the classifier first")

        if x_test.ndim != 2:
            raise ValueError("invalid shape of 'x_test' - should be 2d")

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
            kde = KDE(self.kernel_name).fit(
                self.x_train[mask],
                self.weights_train[mask],
                self.bandwidths[idx],
                self.bandwidth_method,
                **self.kwargs,
            )
            scores[:, idx] = kde.pdf(x_test)

        if np.any(np.all(scores == 0, axis=1)):
            warnings.warn(
                "some labels have been predicted randomly (zero probability issue) - "
                "try again with continuous kernel"
            )

        labels_pred = self.ulabels[np.argmax(self.prior * scores, axis=1)]
        return labels_pred, scores


class KDEOutliersDetection:
    """Outliers detectoion based on kernel density estimation.

    Read more :ref:`here <unconditional_outliers_detection>`.

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data
    >>> m_train, n = 100, 1
    >>> x_train = np.random.normal(0, 1, size=(m_train, n))
    >>> # Fit the outliers detector
    >>> outliers_detector = KDEOutliersDetection("gaussian").fit(x_train)
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
        bandwidth_method: str = "direct_plugin",
        r: float = 0.1,
        **kwargs,
    ):
        """Fit the outliers detector.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n)
            Array containing data points with float type for constructing the detector.
        weights_train : ndarray of shape (m_train,), default=None
            Weights of data points. If None, all data points are equally weighted.
        bandwidth : ndarray of shape (n,), optional
            Smoothing parameter for scaling the estimator. If None, `bandwidth_method`
            is used to compute the `bandwidth`.
        bandwidth_method : {'normal_reference', 'direct_plugin'}, \
                default='normal_reference'
            Name of bandwidth selection method used to compute `bandwidth` when it is
            not given explicitly.
        r : float, default=0.1
            Threshold separating outliers and inliers.

        Returns
        -------
        self : object
            Fitted self instance of KDEOutliersDetection.

        Examples
        --------
        >>> # Prepare data
        >>> m_train, n = 100, 1
        >>> x_train = np.random.normal(0, 1, size=(m_train, n))
        >>> weights_train = np.full((m_train,), 1 / m_train)
        >>> # Fit the outliers detector
        >>> params = (x_train, weights_train)
        >>> outliers_detector = KDEOutliersDetection().fit(*params, r=0.1)
        """
        if r < 0 or r > 1:
            raise ValueError("invalid value of 'r' - should be in range [0, 1]")

        self.kde = KDE(self.kernel_name).fit(
            x_train, weights_train, bandwidth, bandwidth_method, **kwargs
        )
        scores = self.kde.pdf(x_train)
        self.threshold = np.quantile(scores, r)

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        """Predict labels.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Argument of the detector - array containing data points with float type.

        Returns
        -------
        labels_pred : ndarray of shape (m_test,)
            Predicted labels (0 - inlier, 1 - outlier) containing data with int type.

        Examples
        --------
        >>> # Prepare data
        >>> m_train, n = 100, 1
        >>> m_test = 10
        >>> x_train = np.random.normal(0, 1, size=(m_train, n))
        >>> x_test = np.linspace(-3, 3, m_test).reshape(-1, 1)
        >>> # Fit the outliers detector
        >>> outliers_detector = KDEOutliersDetection().fit(x_train, r=0.1)
        >>> # Predict the labels
        >>> labels_pred = outliers_detector.predict(x_test)  # shape: (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the outliers detector first")

        if len(x_test.shape) != 2:
            raise ValueError("invalid shape of 'x_test' - should be 2d")

        scores = self.kde.pdf(x_test)
        labels_pred = np.where(scores <= self.threshold, 1, 0)
        return labels_pred


class KDEClustering:
    """Clustering based on kernel density estimation.

    Read more :ref:`here <unconditional_clustering>`.

    Examples
    --------
    >>> # Prepare data for two clusters
    >>> m_train, n = 100, 1
    >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
    >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> # Fit
    >>> clustering = KDEClustering().fit(x_train)
    """

    def __init__(self):
        self.kernel_name = "gaussian"
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth: Optional[ndarray] = None,
        bandwidth_method: str = "direct_plugin",
        **kwargs,
    ):
        """Fit the model.

        Parameters
        ----------
        x_train : ndarray of shape (m_train, n)
            Array containing data points with float type for constructing the model.
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
            Fitted self instance of KDEClustering.

        Examples
        --------
        >>> # Prepare data for two clusters
        >>> m_train, n = 100, 1
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> weights_train = np.full((m_train,), 1 / m_train)
        >>> # Fit
        >>> clustering = KDEClustering().fit(x_train, weights_train)
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

    def predict(
        self,
        x_test: ndarray,
        algorithm: str = "mean_shift",
        epsilon: float = 1e-8,
        delta: float = 1e-3,  # 1e-1
    ):
        """Predict cluster labels.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n)
            Data points to be grouped - array containing data points with float type.
        algorithm : {'gradient_ascent', 'mean_shift'}, default='mean_shift'
            Name of clustering algorithm.
        epsilon : float, default=1e-8
            Threshold for difference (euclidean distance) of data point position while
            shifting. When the difference is less than epsilon, data point is no longer
            shifted.
        delta : float, default=1e-3
            Acceptance error (euclidean distance) between shifted data point and
            representative of cluster. If the error is less than delta, data point is
            assigned to cluster represented by cluster representative.

        Returns
        -------
        labels_pred : ndarray of shape (m_train,)
            Predicted cluster labels containing data with int type.

        Examples
        --------
        >>> # Prepare data for two clusters
        >>> m_train, n = 100, 1
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n))
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n))
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> # Fit
        >>> clustering = KDEClustering().fit(x_train)
        >>> labels_pred = clustering.predict(x_train)
        """
        if not self.fitted:
            raise RuntimeError("fit the clusterer first")

        if algorithm == "gradient_ascent":
            x_k = gradient_ascent(
                self.x_train,
                self.weights_train,
                x_test,
                self.bandwidth,
                epsilon,
            )
        elif algorithm == "mean_shift":
            x_k = mean_shift(
                self.x_train,
                self.weights_train,
                x_test,
                self.bandwidth,
                epsilon,
            )
        else:
            raise ValueError("invalid 'algorithm'")

        labels_pred = assign_labels(x_k, delta)
        return labels_pred
