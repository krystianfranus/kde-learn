import numpy as np


def uniform(x):
    return np.where(np.abs(x) <= 1, 0.5, 0)


def gaussian(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)


def epanechnikov(x):
    return np.where(np.abs(x) <= 1, 3 / 4 * (1 - x ** 2), 0)


def cauchy(x):
    return 2 / (np.pi * (x ** 2 + 1) ** 2)
