import numpy as np

from .kernels import gaussian


def estimate_bandwidth(x_train, kernel='gaussian'):
    m_train = x_train.shape[0]
    std_x = np.std(x_train, axis=0, ddof=1)
    if m_train == 1:
        std_x = np.std(x_train, axis=0)
    std_x[std_x == 0] = 1

    if kernel == 'gaussian':
        W, U = 1 / (2 * np.sqrt(np.pi)), 1
    elif kernel == 'uniform':
        W, U = 1 / 2, 1 / 3
    elif kernel == 'cauchy':
        W, U = 5 / (4 * np.pi), 1
    elif kernel == 'epanechnikov':
        W, U = 0.6, 0.2
    else:
        raise ValueError(f'invalid kernel: {kernel}')
    WU2 = W / (U ** 2)
    Z = 3 / (8 * np.sqrt(np.pi))

    bandwidth = (WU2 / (Z * m_train)) ** (1 / 5) * std_x
    return bandwidth


class Kde():
    def fit(self, x_train, weights_train=None, bandwidth=None):
        self.m_train, self.n = x_train.shape
        self.x_train = np.copy(x_train)

        if weights_train is None:
            self.weights_train = np.full(self.m_train, 1 / self.m_train)
        else:
            self.weights_train = np.copy(weights_train)
            self.weights_train = self.weights_train / self.weights_train.sum()

        if bandwidth is None:
            self.bandwidth = estimate_bandwidth(self.x_train)
        elif bandwidth.any() <= 0:
            raise ValueError(f'bandwidth needs to be greater than zero or None, got {bandwidth}')
        else:
            self.bandwidth = np.copy(bandwidth)

        self.s = np.ones(self.m_train)
        return self

    def score_samples(self, x_test):
        scores = 1 / (np.prod(self.bandwidth)) * np.sum((self.weights_train / (self.s ** self.n))[:, None] * np.prod(
            gaussian((x_test - self.x_train[:, None]) / (self.bandwidth * self.s[:, None, None])), axis=2), axis=0)
        return scores
