from typing import Optional

import numpy as np
from numpy import ndarray

from kdelearn.cutils import compute_ckde

from .utils import scotts_rule


class CKDE:
    """Conditional kernel density estimator."""

    def __init__(self, kernel_name: str = "gaussian"):
        self.kernel_name = kernel_name

    def fit(
        self,
        x_train: ndarray,
        w_train: ndarray,
        weights_train: Optional[ndarray] = None,
        bandwidth_x: Optional[ndarray] = None,
        bandwidth_w: Optional[ndarray] = None,
    ):
        self.x_train = x_train
        self.w_train = w_train

        if weights_train is None:
            m_train = self.x_train.shape[0]
            self.weights_train = np.full(m_train, 1 / m_train)
        else:
            self.weights_train = weights_train / weights_train.sum()

        if bandwidth_x is None and bandwidth_w is None:
            self.bandwidth_x = scotts_rule(self.x_train, self.kernel_name)
            self.bandwidth_w = scotts_rule(self.w_train, self.kernel_name)
        else:
            self.bandwidth_x = bandwidth_x
            self.bandwidth_w = bandwidth_w

        return self

    def pdf(self, x_test: ndarray, w_test: ndarray) -> ndarray:
        """Compute estimation of conditional probability density function."""
        scores = compute_ckde(
            self.x_train,
            self.w_train,
            x_test,
            w_test,
            self.weights_train,
            self.bandwidth_x,
            self.bandwidth_w,
            self.kernel_name,
        )
        return scores

    # TODO: https://stats.stackexchange.com/questions/43674/simple-sampling-method-for-a-kernel-density-estimator
    def sample(self):
        raise NotImplementedError
