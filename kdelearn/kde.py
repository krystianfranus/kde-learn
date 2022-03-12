from typing import Optional

import numpy as np
from numpy import ndarray

from kdelearn.cutils import compute_kde

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
        kde = Kde(kernel_name=kernel_name).fit(
            x_train[mask],
            weights_train=weights,
            bandwidth=bandwidth,
        )
        scores[:, idx] = kde.score_samples(x_test)

    labels_pred = ulabels[np.argmax(scores * prior, axis=1)]
    return labels_pred


class Kde:
    def __init__(self, kernel_name: str = "gaussian"):
        self.kernel_name = kernel_name

    def fit(
        self,
        x_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth: Optional[ndarray] = None,
    ):
        if len(x_train.shape) != 2:
            raise RuntimeError("x_train must be 2d ndarray")
        self.x_train = np.copy(x_train)

        if weights_train is None:
            m_train = self.x_train.shape[0]
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            if len(weights_train.shape) != 1:
                raise RuntimeError("weights_train must be 1d ndarray")
            if not (weights_train > 0).all():
                raise ValueError("weights_train must be positive")
            self.weights_train = np.copy(weights_train)
            self.weights_train = self.weights_train / self.weights_train.sum()

        if bandwidth is None:
            self.bandwidth = estimate_bandwidth(
                self.x_train,
                self.kernel_name,
            )
        else:
            if not (bandwidth > 0).all():
                raise ValueError("bandwidth must be positive")
            self.bandwidth = np.copy(bandwidth)

        return self

    def score_samples(self, x_test: ndarray) -> ndarray:
        scores = compute_kde(
            self.x_train,
            x_test,
            self.weights_train,
            self.bandwidth,
            self.kernel_name,
        )
        return scores
