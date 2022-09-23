from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .bandwidth_selection import direct_plugin, ml_cv, normal_reference, ste_plugin
from .cutils import compute_d, compute_kde


class CKDE:
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
        **kwargs,
    ):
        self.x_train = x_train
        self.w_train = w_train
        self.w_star = w_star
        self.m_train = self.x_train.shape[0]
        self.n_x = self.x_train.shape[1]
        self.n_w = self.w_train.shape[1]

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
                bandwidth = ml_cv(z_train, self.kernel_name, self.weights_train)
            else:
                raise ValueError("invalid 'bandwidth_method'")
            self.bandwidth_x = bandwidth[: self.n_x]
            self.bandwidth_w = bandwidth[self.n_x :]
        else:
            self.bandwidth_x = bandwidth_x
            self.bandwidth_w = bandwidth_w

        self.fitted = True
        return self

    def pdf(
        self,
        x_test: ndarray,
    ) -> Tuple[ndarray, ndarray]:
        d = compute_d(
            self.w_train,
            self.weights_train,
            self.w_star,
            self.bandwidth_w,
            self.kernel_name,
        )
        scores = compute_kde(
            self.x_train,
            d,
            x_test,
            self.bandwidth_x,
            self.kernel_name,
        )
        return scores, d
