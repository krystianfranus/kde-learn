import warnings
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .bandwidth_selection import direct_plugin, kernel_properties, normal_reference
from .ckde import CKDE
from .cutils import assign_labels, compute_d, gradient_ascent, mean_shift


class CKDEClassification:
    """Bayes' classifier based on conditional kernel density estimation.

    TODO: <MATH FORMULA and READ MORE and REFERENCES>

    Parameters
    ----------
    kernel_name : {'gaussian', 'uniform', 'epanechnikov', 'cauchy'}, default='gaussian'
        Name of kernel function.

    Examples
    --------
    >>> # Prepare data for two classes
    >>> m_train = 100
    >>> n_x, n_y = 1, 1
    >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
    >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
    >>> labels_train1 = np.full(m_train // 2, 1)
    >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
    >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
    >>> labels_train2 = np.full(m_train // 2, 2)
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> y_train = np.concatenate((y_train1, y_train2))
    >>> labels_train = np.concatenate((labels_train1, labels_train2))
    >>> y_star = np.array([0.0] * n_y)
    >>> # Fit
    >>> classifier = CKDEClassification().fit(x_train, y_train, y_star, labels_train)
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
        labels_train: ndarray,
        weights_train: Optional[ndarray] = None,
        share_bandwidth: bool = False,
        bandwidths_x: Optional[ndarray] = None,
        bandwidths_y: Optional[ndarray] = None,
        bandwidth_method: str = "direct_plugin",
        prior_prob: Optional[ndarray] = None,
        **kwargs,
    ):
        """Fit the classifier.

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
        labels_train : ndarray of shape (m_train,)
            Labels of data points as an array containing data with int type.
        weights_train : ndarray of shape (m_train,), default=None
            Weights for data points. If None, all points are equally weighted.
        share_bandwidth : bool, default=False
            Determines whether all classes should have common bandwidth.
            If False, estimator of each class gets its own bandwidth.
        bandwidths_x : ndarray of shape (n_classes, n_x), optional
            Smoothing parameter of describing variables for each class.
        bandwidths_y : ndarray of shape (n_classes, n_y), optional
            Smoothing parameter of conditioning variables for each class.
        bandwidth_method : {'normal_reference', 'direct_plugin', 'ste_plugin', \
                'ml_cv'}, default='normal_reference'
            Name of bandwidth selection method used to compute smoothing parameter.
        prior_prob : ndarray of shape (n_classes,), default=None
            Prior probabilities of each class. If None, all classes are equally
            probable.

        Returns
        -------
        self : object
            Fitted self instance of CKDEClassification.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
        >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> labels_train1 = np.full(m_train // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
        >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> labels_train2 = np.full(m_train // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> y_train = np.concatenate((y_train1, y_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> y_star = np.array([0.0] * n_y)
        >>> weights_train = np.random.uniform(0, 1, size=(m_train,))
        >>> # Fit
        >>> prior_prob = np.array([0.3, 0.7])
        >>> params = (x_train, y_train, y_star, labels_train, weights_train)
        >>> classifier = CKDEClassification().fit(*params, prior_prob=prior_prob)
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of 'x_train' - should be 2d")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]
        n_x = self.x_train.shape[1]

        if y_train.ndim != 2:
            raise ValueError("invalid shape of 'y_train' - should be 2d")
        if y_train.shape[0] != x_train.shape[0]:
            raise ValueError("invalid size of 'y_train'")
        self.y_train = y_train
        n_y = self.y_train.shape[1]

        if y_star.ndim != 1:
            raise ValueError("invalid shape of 'y_star' - should be 1d")
        if y_star.shape[0] != n_y:
            raise ValueError(f"invalid size of 'y_star'- should contain {n_y} values")
        self.y_star = y_star

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
            if not (weights_train > 0).all():
                raise ValueError("'weights_train' must be positive")
            self.weights_train = weights_train / weights_train.sum()

        self.ulabels = np.unique(labels_train)  # Sorted unique labels
        self.n_classes = self.ulabels.shape[0]

        self.bandwidth_method = bandwidth_method
        if share_bandwidth:
            if self.bandwidth_method == "normal_reference":
                bandwidth_y = normal_reference(y_train, weights_train, self.kernel_name)
                bandwidth_x = normal_reference(
                    self.x_train, weights_train, self.kernel_name
                )
            elif self.bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                bandwidth_y = direct_plugin(
                    y_train, weights_train, self.kernel_name, stage
                )
                bandwidth_x = direct_plugin(
                    self.x_train, weights_train, self.kernel_name, stage
                )
            else:
                raise ValueError("invalid 'bandwidth_method'")
            self.bandwidths_x = np.full((self.n_classes, n_x), bandwidth_x)
            self.bandwidths_y = np.full((self.n_classes, n_y), bandwidth_y)
        else:
            if bandwidths_x is not None and bandwidths_y is not None:
                self.bandwidths_x = bandwidths_x
                self.bandwidths_y = bandwidths_y
            else:
                self.bandwidths_x = np.full((self.n_classes,), bandwidths_x)
                self.bandwidths_y = np.full((self.n_classes,), bandwidths_y)

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
        x_test : ndarray of shape (m_test, n_x)
            Grid data points (describing variables) as an array containing data with
            float type.

        Returns
        -------
        labels_pred : ndarray of shape (m_test,)
            Predicted labels as an array containing data with int type.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> m_test = 10
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
        >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> labels_train1 = np.full(m_train // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
        >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> labels_train2 = np.full(m_train // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> y_train = np.concatenate((y_train1, y_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> y_star = np.array([0.0] * n_y)
        >>> # Fit the classifier
        >>> x_test = np.random.uniform(-1, 4, size=(m_test, n_x))
        >>> params = (x_train, y_train, y_star, labels_train)
        >>> classifier = CKDEClassification().fit(*params)
        >>> # Predict labels
        >>> labels_pred = classifier.predict(x_test)  # labels_pred shape (10,)
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
        x_test : ndarray of shape (m_test, n_x)
            Grid data points (describing variables) as an array containing data with
            float type.

        Returns
        -------
        scores : ndarray of shape (m_test, n_classes)
            Predicted scores as an array containing data with float type.

        Examples
        --------
        >>> # Prepare data for two classes
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> m_test = 10
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
        >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> labels_train1 = np.full(m_train // 2, 1)
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
        >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> labels_train2 = np.full(m_train // 2, 2)
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> y_train = np.concatenate((y_train1, y_train2))
        >>> labels_train = np.concatenate((labels_train1, labels_train2))
        >>> y_star = np.array([0.0] * n_y)
        >>> # Fit the classifier
        >>> x_test = np.random.uniform(-1, 4, size=(m_test, n_x))
        >>> params = (x_train, y_train, y_star, labels_train)
        >>> classifier = CKDEClassification().fit(*params)
        >>> # Compute pdf of each class
        >>> scores = classifier.pdfs(x_test)  # scores shape (10, 2)
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
            ckde = CKDE(self.kernel_name).fit(
                self.x_train[mask],
                self.y_train[mask],
                self.y_star,
                self.weights_train[mask],
                self.bandwidths_x[idx],
                self.bandwidths_y[idx],
                self.bandwidth_method,
                **self.kwargs,
            )
            scores[:, idx] = ckde.pdf(x_test)

        if np.any(np.all(scores == 0, axis=1)):
            warnings.warn(
                "some labels have been predicted randomly (zero probability issue) - "
                "try again with continuous kernel"
            )

        labels_pred = self.ulabels[np.argmax(self.prior * scores, axis=1)]
        return labels_pred, scores


class CKDEOutliersDetection:
    """Outliers detection based on conditional kernel density estimation.

    TODO: <READ MORE>

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
    >>> # Fit the outliers detector
    >>> params = (x_train, y_train, y_star)
    >>> outliers_detector = CKDEOutliersDetection("gaussian").fit(*params)
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
        bandwidth_method: str = "direct_plugin",
        r: float = 0.1,
        **kwargs,
    ):
        """Fit the outliers detector.

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
        weights_train : ndarray of shape (m_train,), default=None
            Weights for data points. If None is passed, all points are equally weighted.
        bandwidth_x : ndarray of shape (n_x,), optional
            Smoothing parameter of describing variables.
        bandwidth_y : ndarray of shape (n_y,), optional
            Smoothing parameter of conditioning variables.
        bandwidth_method : {'normal_reference', 'direct_plugin'}, \
                default='normal_reference'
            Name of bandwidth selection method used to compute smoothing parameter
            when `bandwidth` is not given explicitly.
        r : float, default=0.1
            Threshold separating outliers and inliers.

        Returns
        -------
        self : object
            Fitted self instance of CKDEOutliersDetection.

        Examples
        --------
        >>> # Prepare data
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> x_train = np.random.normal(0, 1, size=(m_train, n_x))
        >>> y_train = np.random.normal(0, 1, size=(m_train, n_y))
        >>> y_star = np.array([0.0] * n_y)
        >>> weights_train = np.random.uniform(0, 1, size=(m_train,))
        >>> # Fit the outliers detector
        >>> params = (x_train, y_train, y_star, weights_train)
        >>> outliers_detector = CKDEOutliersDetection().fit(*params, r=0.1)
        """
        if r < 0 or r > 1:
            raise ValueError("invalid value of 'r' - should be in range [0, 1]")

        self.ckde = CKDE(self.kernel_name).fit(
            x_train,
            y_train,
            y_star,
            weights_train,
            bandwidth_x,
            bandwidth_y,
            bandwidth_method,
            **kwargs,
        )
        scores = self.ckde.pdf(x_train)

        idx_sorted = np.argsort(scores)
        scores_ord = scores[idx_sorted]
        cond_weights_train_ord = self.ckde.cond_weights_train[idx_sorted]

        cond_weights_train_ord_cumsum = np.cumsum(cond_weights_train_ord)
        k = np.where(cond_weights_train_ord_cumsum > r)[0][0] - 1

        tmp1 = (cond_weights_train_ord_cumsum[k + 1] - r) * scores_ord[k]
        tmp2 = (r - cond_weights_train_ord_cumsum[k]) * scores_ord[k + 1]
        self.threshold = (tmp1 + tmp2) / cond_weights_train_ord[k + 1]
        # self.threshold = np.quantile(scores, r)

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        """Predict the labels.

        Parameters
        ----------
        x_test : ndarray of shape (m_test, n_x)
            Grid data points (describing variables) as a 2D array containing data with
            float type.

        Returns
        -------
        labels_pred : ndarray of shape (m_test,)
            Predicted labels (0 - inlier, 1 - outlier) as an array containing data
            with int type.

        Examples
        --------
        >>> # Prepare data
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> m_test = 10
        >>> x_train = np.random.normal(0, 1, size=(m_train, n_x))
        >>> y_train = np.random.normal(0, 1, size=(m_train, n_y))
        >>> y_star = np.array([0.0] * n_y)
        >>> x_test = np.random.uniform(-3, 3, size=(m_test, n_x))
        >>> # Fit the outliers detector
        >>> params = (x_train, y_train, y_star)
        >>> outliers_detector = CKDEOutliersDetection().fit(*params, r=0.1)
        >>> # Predict the labels
        >>> labels_pred = outliers_detector.predict(x_test)  # labels_pred shape (10,)
        """
        if not self.fitted:
            raise RuntimeError("fit the outliers detector first")

        if len(x_test.shape) != 2:
            raise ValueError("invalid shape of 'x_test' - should be 2d")

        scores = self.ckde.pdf(x_test)
        labels_pred = np.where(scores <= self.threshold, 1, 0)
        return labels_pred


class CKDEClustering:
    """Clustering based on conditional kernel density estimation.

    TODO: <READ MORE>

    Examples
    --------
    >>> # Prepare data for two clusters
    >>> m_train = 100
    >>> n_x, n_y = 1, 1
    >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
    >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
    >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
    >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> y_train = np.concatenate((y_train1, y_train2))
    >>> y_star = np.array([0.0] * n_y)
    >>> # Fit
    >>> clustering = CKDEClustering().fit(x_train, y_train, y_star)
    """

    def __init__(self):
        self.kernel_name = "gaussian"
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        y_train: ndarray,
        y_star: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth_x: Optional[ndarray] = None,
        bandwidth_y: Optional[ndarray] = None,
        bandwidth_method: str = "direct_plugin",
        **kwargs,
    ):
        """Fit the model.

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
            Fitted self instance of CKDEClustering.

        Examples
        --------
        >>> # Prepare data for two clusters
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
        >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
        >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> y_train = np.concatenate((y_train1, y_train2))
        >>> y_star = np.array([0.0] * n_y)
        >>> weights_train = np.random.uniform(0, 1, size=(m_train,))
        >>> # Fit
        >>> clustering = CKDEClustering().fit(x_train, y_train, y_star, weights_train)
        """
        if x_train.ndim != 2:
            raise ValueError("invalid shape of 'x_train' - should be 2d")
        self.x_train = x_train
        self.m_train = self.x_train.shape[0]
        self.n_x = self.x_train.shape[1]

        if y_train.ndim != 2:
            raise ValueError("invalid shape of 'y_train' - should be 2d")
        if y_train.shape[0] != self.x_train.shape[0]:
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
                bandwidth = direct_plugin(z_train, self.kernel_name, stage)
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
        x_test : ndarray of shape (m_test, n_x)
            Grid data points (describing variables) as an array containing data with
            float type.
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
            Predicted labels as an array containing data with int type.

        Examples
        --------
        >>> # Prepare data for two clusters
        >>> m_train = 100
        >>> n_x, n_y = 1, 1
        >>> x_train1 = np.random.normal(0, 1, size=(m_train // 2, n_x))
        >>> y_train1 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> x_train2 = np.random.normal(3, 1, size=(m_train // 2, n_x))
        >>> y_train2 = np.random.normal(0, 1, size=(m_train // 2, n_y))
        >>> x_train = np.concatenate((x_train1, x_train2))
        >>> y_train = np.concatenate((y_train1, y_train2))
        >>> y_star = np.array([0.0] * n_y)
        >>> # Fit
        >>> clustering = CKDEClustering().fit(x_train, y_train, y_star)
        >>> labels_pred = clustering.predict()
        """
        cond_weights_train = compute_d(
            self.y_train,
            self.weights_train,
            self.y_star,
            self.bandwidth_y,
            self.kernel_name,
        )

        if algorithm == "gradient_ascent":
            x_k = gradient_ascent(
                self.x_train, cond_weights_train, x_test, self.bandwidth_x, epsilon
            )
        elif algorithm == "mean_shift":
            x_k = mean_shift(
                self.x_train, cond_weights_train, x_test, self.bandwidth_x, epsilon
            )
        else:
            raise ValueError("invalid 'algorithm'")
        labels_pred = assign_labels(x_k, delta)

        return labels_pred
