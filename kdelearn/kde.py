import numpy as np

from .kernels import gaussian
from .utils import estimate_bandwidth


class Kde:

    def __init__(self):
        self.x_train = None
        self.m_train = None
        self.n = None
        self.weights_train = None
        self.bandwidth = None
        self.s = None
        self.kernel = gaussian

    def fit(self, x_train, weights_train=None, bandwidth=None):
        self.x_train = np.copy(x_train)
        self.m_train, self.n = self.x_train.shape

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            self.weights_train = np.copy(weights_train)
            self.weights_train = self.weights_train / self.weights_train.sum()

        if bandwidth is None:
            self.bandwidth = estimate_bandwidth(self.x_train)
        else:
            assert bandwidth.any() > 0, f'Bandwidth needs to be greater than zero. Got {bandwidth}.'
            self.bandwidth = np.copy(bandwidth)

        self.s = np.ones(self.m_train)
        return self

    def score_samples(self, x_test):
        kernel_values = self.kernel((x_test - self.x_train[:, None]) / (self.bandwidth * self.s[:, None, None]))
        scores = 1 / (np.prod(self.bandwidth)) * np.sum(
            (self.weights_train / (self.s ** self.n))[:, None] * np.prod(kernel_values, axis=2), axis=0)
        return scores
