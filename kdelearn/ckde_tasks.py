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
from .ckde import CKDE
from .cutils import assign_labels, compute_d, gradient_ascent, mean_shift


class CKDEClassification:
    def __init__(self, kernel_name: str = "gaussian"):
        if kernel_name not in kernel_properties:
            available_kernels = list(kernel_properties.keys())
            raise ValueError(f"invalid 'kernel_name' - try one of {available_kernels}")
        self.kernel_name = kernel_name
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        w_train: ndarray,
        w_star: ndarray,
        labels_train: ndarray,
        weights_train: Optional[ndarray] = None,
        share_bandwidth: bool = False,
        bandwidth_method: str = "normal_reference",
        prior_prob: Optional[ndarray] = None,
        **kwargs,
    ):
        self.x_train = x_train
        self.w_train = w_train
        self.m_train = self.x_train.shape[0]
        self.n_x = self.x_train.shape[1]
        self.n_w = self.w_train.shape[1]

        self.labels_train = labels_train

        self.w_star = w_star

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            self.weights_train = weights_train / weights_train.sum()

        self.bandwidth_x = None
        self.bandwidth_w = None
        self.bandwidth_method = bandwidth_method

        if share_bandwidth:
            z_train = np.concatenate((self.x_train, self.w_train), axis=1)
            if self.bandwidth_method == "normal_reference":
                bandwidth = normal_reference(z_train, self.kernel_name)
            elif self.bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                bandwidth = direct_plugin(z_train, self.kernel_name, stage)
            elif self.bandwidth_method == "ste_plugin":
                bandwidth = ste_plugin(z_train, self.kernel_name)
            elif self.bandwidth_method == "ml_cv":
                bandwidth = ml_cv(z_train, self.kernel_name, self.weights_train)
            else:
                raise ValueError("invalid 'bandwidth_method'")
            self.bandwidth_x = bandwidth[: self.n_x]
            self.bandwidth_w = bandwidth[self.n_x :]

        self.ulabels = np.unique(labels_train)  # Sorted unique labels
        self.n_classes = self.ulabels.shape[0]
        if prior_prob is None:
            self.prior = self._compute_prior()
        else:
            self.prior = prior_prob / prior_prob.sum()

        self.kwargs = kwargs

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        labels_pred, _ = self._classify(x_test)
        return labels_pred

    def pdfs(self, x_test: ndarray) -> ndarray:
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
                self.w_train[mask],
                self.w_star,
                self.weights_train[mask],
                self.bandwidth_x,
                self.bandwidth_w,
                self.bandwidth_method,
                **self.kwargs,
            )
            scores[:, idx], _ = ckde.pdf(x_test)

        labels_pred = self.ulabels[np.argmax(self.prior * scores, axis=1)]
        return labels_pred, scores


class CKDEOutliersDetection:
    def __init__(self, kernel_name: str = "gaussian"):
        self.kernel_name = kernel_name
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        w_train: ndarray,
        w_star: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth_x: Optional[ndarray] = None,
        bandwidth_w: Optional[ndarray] = None,
        bandwidth_method: str = "normal_reference",
        r: float = 0.1,
        **kwargs,
    ):
        self.m_train = x_train.shape[0]
        self.n_w = w_train.shape[1]

        self.ckde = CKDE(self.kernel_name).fit(
            x_train,
            w_train,
            w_star,
            weights_train,
            bandwidth_x,
            bandwidth_w,
            bandwidth_method,
            **kwargs,
        )
        scores, d = self.ckde.pdf(x_train)

        idx_sorted = np.argsort(scores)
        scores_ord = scores[idx_sorted]
        d_ord = d[idx_sorted]

        d_ord_cumsum = np.cumsum(d_ord)
        k = np.where(d_ord_cumsum > r)[0][0] - 1

        tmp1 = (d_ord_cumsum[k + 1] - r) * scores_ord[k]
        tmp2 = (r - d_ord_cumsum[k]) * scores_ord[k + 1]
        self.threshold = (tmp1 + tmp2) / d_ord[k + 1]
        # self.threshold = np.quantile(scores, r)

        self.fitted = True
        return self

    def predict(self, x_test: ndarray) -> ndarray:
        scores, _ = self.ckde.pdf(x_test)
        labels_pred = np.where(scores <= self.threshold, 1, 0)
        return labels_pred


class CKDEClustering:
    def __init__(self):
        self.kernel_name = "gaussian"
        self.fitted = False

    def fit(
        self,
        x_train: ndarray,
        w_train: ndarray,
        w_star: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth_x: Optional[ndarray] = None,
        bandwidth_w: Optional[ndarray] = None,
        bandwidth_method: str = "normal_reference",
        **kwargs,
    ):
        self.x_train = x_train
        self.w_train = w_train
        self.w_star = w_star
        self.m_train = self.x_train.shape[0]
        self.n_x = self.x_train.shape[1]

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            self.weights_train = weights_train / weights_train.sum()

        if bandwidth_x is None:
            z_train = np.concatenate((self.x_train, self.w_train), axis=1)
            if bandwidth_method == "normal_reference":
                bandwidth = normal_reference(z_train, self.kernel_name)
            elif bandwidth_method == "direct_plugin":
                stage = kwargs["stage"] if "stage" in kwargs else 2
                bandwidth = direct_plugin(z_train, self.kernel_name, stage)
            elif bandwidth_method == "ste_plugin":
                bandwidth = ste_plugin(z_train, self.kernel_name)
            elif bandwidth_method == "ml_cv":
                bandwidth = ml_cv(z_train, self.kernel_name)
            else:
                raise ValueError("invalid 'bandwidth_method'")
            self.bandwidth_x = bandwidth[: self.n_x]
            self.bandwidth_w = bandwidth[self.n_x :]
        else:
            self.bandwidth_x = bandwidth_x
            self.bandwidth_w = bandwidth_w

        self.fitted = True
        return self

    def predict(
        self,
        algorithm: str = "mean_shift",
        epsilon: float = 1e-8,
        delta: float = 1e-3,  # 1e-1
    ):
        d = compute_d(
            self.w_train,
            self.weights_train,
            self.w_star,
            self.bandwidth_w,
            self.kernel_name,
        )

        if algorithm == "gradient_ascent":
            x_k = gradient_ascent(self.x_train, d, self.bandwidth_x, epsilon)
        elif algorithm == "mean_shift":
            x_k = mean_shift(self.x_train, d, self.bandwidth_x, epsilon)
        else:
            raise ValueError("invalid 'algorithm'")
        labels_pred = assign_labels(x_k, delta)

        return labels_pred
