from typing import Optional

import numpy as np
from numpy import ndarray

from kdelearn.cutils import compute_kde

from .utils import estimate_bandwidth


class Kde:
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name

    def fit(self, x_train: ndarray, weights_train: Optional[ndarray] = None):
        if len(x_train.shape) != 2:
            raise RuntimeError("x_train must be 2d ndarray")
        self.x_train = np.copy(x_train)
        m_train = self.x_train.shape[0]

        # TODO: check whether weights are positive numbers
        if weights_train is None:
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            if len(weights_train.shape) != 1:
                raise RuntimeError("weights_train must be 1d ndarray")
            if not (weights_train > 0).all():
                raise RuntimeError("weights_train must be positive")
            self.weights_train = np.copy(weights_train)
            self.weights_train = self.weights_train / self.weights_train.sum()

        self.bandwidth = estimate_bandwidth(
            self.x_train,
            kernel_name=self.kernel_name,
        )
        return self

    def score_samples(self, x_test: ndarray) -> ndarray:
        scores = compute_kde(
            self.x_train,
            self.weights_train,
            self.bandwidth,
            x_test,
            self.kernel_name,
        )
        return scores
