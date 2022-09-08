import numpy as np
from numpy import ndarray


def accuracy_loo(
    x_train: ndarray,
    labels_train: ndarray,
    model,
    **kwargs,
) -> float:
    """Leave-one-out accuracy - ratio of correctly classified data points based on
    leave-one-out approach.

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

    Examples
    --------
    >>> # Prepare data for two classes
    >>> x_train1 = np.random.normal(0, 1, size=(100 // 2, 1))
    >>> labels_train1 = np.full(100 // 2, 1)
    >>> x_train2 = np.random.normal(3, 1, size=(100 // 2, 1))
    >>> labels_train2 = np.full(100 // 2, 2)
    >>> x_train = np.concatenate((x_train1, x_train2))
    >>> labels_train = np.concatenate((labels_train1, labels_train2))
    >>> # Classify and compute accuracy
    >>> model = KDEClassification()
    >>> accuracy = accuracy_loo(x_train, labels_train, model)
    """
    if x_train.ndim != 2:
        raise ValueError("invalid shape of 'x_train' - should be 2d")
    m_train = x_train.shape[0]

    if labels_train.ndim != 1:
        raise ValueError("invalid shape of 'labels_train' - should be 1d")
    if not np.issubdtype(labels_train.dtype, np.integer):
        raise ValueError("invalid dtype of 'labels_train' - should be of int type")

    if not hasattr(model, "fit"):
        raise AttributeError(f"'{model}' object has no attribute 'fit'")
    if not hasattr(model, "predict"):
        raise AttributeError(f"'{model}' object has no attribute 'predict'")

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
