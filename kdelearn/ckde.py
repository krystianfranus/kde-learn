import numpy as np

from .kernels import gaussian
from .utils import estimate_bandwidth


class Ckde:

    def __init__(self):
        self.kernel = gaussian

    def fit(self, y_train, w_train, bandwidth=None):
        self.m_train = y_train.shape[0]
        self.n_y, self.n_w = y_train.shape[1], w_train.shape[1]
        self.y_train = np.copy(y_train)
        self.w_train = np.copy(w_train)

        if bandwidth is None:
            x_train = np.concatenate((self.y_train, self.w_train), axis=1)
            self.bandwidth = estimate_bandwidth(x_train)
            self.bandwidth_y = self.bandwidth[:self.n_y]
            self.bandwidth_w = self.bandwidth[self.n_y:]
        else:
            assert bandwidth.any() > 0, f'Bandwidth needs to be greater than zero. Got {bandwidth}.'
            self.bandwidth_y = bandwidth[:self.n_y]
            self.bandwidth_w = bandwidth[self.n_y:]

        self.s = np.ones(self.m_train)
        return self

    def score_samples(self, y_test, w_star):
        kernel_w_values = self.kernel((w_star - self.w_train[:, None]) / (self.bandwidth_w * self.s[:, None, None]))
        self.d = np.prod(kernel_w_values, axis=2)
        self.d = self.m_train * self.d / np.sum(self.d, axis=0)

        kernel_y_values = self.kernel((y_test - self.y_train[:, None]) / (self.bandwidth_y * self.s[:, None, None]))
        scores = 1 / (self.m_train * np.prod(self.bandwidth_y)) * np.sum(
            (self.d / (self.s[:, None] ** self.n_y)) * np.prod(kernel_y_values, axis=2), axis=0)
        return scores
