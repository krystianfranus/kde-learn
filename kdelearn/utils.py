from numpy import ndarray


def accuracy(labels_true: ndarray, labels_pred: ndarray) -> float:
    """Accuracy score computes fraction of correctly classified samples.

    Parameters
    ----------
    labels_true : ndarray
        True (ground truth) labels as an array containing data with int type.
    labels_pred : ndarray
        Predicted labels returned by classifier as an array containing data with int
        type.

    Examples
    --------
    >>> labels_true = np.array([0, 1])
    >>> labels_pred = np.array([1, 1])
    >>> accuracy(labels_true, labels_pred)

    Returns
    -------
    accuracy : float
        Fraction of correctly classified samples.
    """
    if labels_true.size != labels_pred.size:
        raise RuntimeError("Both arrays must be of the same size")
    return (labels_true == labels_pred).sum() / labels_true.size
