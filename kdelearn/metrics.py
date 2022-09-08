import numpy as np
from numpy import ndarray


def accuracy_loo(
    x_train: ndarray,
    labels_train: ndarray,
    model,
    **kwargs,
) -> float:
    """Leave-one-out accuracy.

    Ratio of correctly classified data points based on leave-one-out approach.

    Parameters
    ----------
    x_train : ndarray of shape (m_train, n)
        Data points as an array containing data with float type.
    labels_train : ndarray of shape (m_train,)
        Labels of data points as an array containing data with int type.
    model
        Classifier with defined `fit` and `predict` methods.

    Returns
    -------
    accuracy : float
        Leave-one-out accuracy.
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of x_train - should be 2d")
    m_train = x_train.shape[0]

    if labels_train.ndim != 1:
        raise ValueError("invalid shape of labels_train - should be 1d")
    if not np.issubdtype(labels_train.dtype, np.integer):
        raise ValueError("invalid dtype of labels_train - should be of int type")

    labels_pred = np.empty((m_train,), dtype=np.int32)
    for i in range(m_train):
        mask = np.delete(np.arange(m_train), i)
        classifier = model.fit(
            x_train[mask],
            labels_train[mask],
            **kwargs,
        )
        labels_pred[i] = classifier.predict(x_train[i : i + 1])

    accuracy = np.sum(labels_train == labels_pred) / m_train
    return accuracy
