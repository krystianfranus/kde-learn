import numpy as np
import pytest

from kdelearn.kde_tasks import KDEClassification


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
@pytest.mark.parametrize("share_bandwidth", [False, True])
@pytest.mark.parametrize("bandwidth_method", ["normal_reference", "direct_plugin"])
def test_kde_classifier(
    classification_data,
    kernel_name,
    share_bandwidth,
    bandwidth_method,
):
    x_train, labels_train, weights_train = classification_data
    classifier = KDEClassification(kernel_name)
    classifier = classifier.fit(
        x_train,
        labels_train,
        weights_train=weights_train,
        bandwidth_method=bandwidth_method,
        share_bandwidth=share_bandwidth,
    )
    labels_pred = classifier.predict(x_train)
    assert classifier.fitted
    assert classifier.weights_train.ndim == 1
    assert (classifier.weights_train > 0).all()
    assert (classifier.prior > 0).all()
    assert labels_pred.shape[0] == x_train.shape[0]
    assert labels_pred.shape[0] == labels_train.shape[0]
    assert labels_pred.ndim == labels_train.ndim
    assert labels_pred.ndim == 1


def test_kde_classifier_with_fixed_bandwidth(classification_data):
    x_train, labels_train, _ = classification_data
    n = x_train.shape[1]
    n_classes = np.unique(labels_train).shape[0]
    bandwidths = np.array([0.5] * n_classes * n).reshape(n_classes, n)
    classifier = KDEClassification()
    classifier.fit(x_train, labels_train, bandwidths=bandwidths)
    assert classifier.fitted
    assert classifier.bandwidths.ndim == 2
    assert (classifier.bandwidths > 0).all()


def test_kde_classifier_with_fixed_prior_prob(classification_data):
    x_train, labels_train, _ = classification_data
    n_classes = np.unique(labels_train).shape[0]
    prior_prob = np.array([0.5] * n_classes)
    classifier = KDEClassification()
    classifier.fit(x_train, labels_train, prior_prob=prior_prob)
    assert classifier.fitted
    assert classifier.prior.ndim == 1
    assert classifier.prior.shape[0] == n_classes
    assert (classifier.prior > 0).all()


def test_kde_classifier_pdfs(classification_data):
    x_train, labels_train, _ = classification_data
    n_classes = np.unique(labels_train).shape[0]
    classifier = KDEClassification()
    scores = classifier.fit(x_train, labels_train).pdfs(x_train)
    assert classifier.fitted
    assert scores.ndim == 2
    assert scores.shape[1] == n_classes
    assert scores.shape[0] == x_train.shape[0]
    assert (scores >= 0).all()
    assert (scores < 1).all()


def test_kde_classifier_invalid():
    with pytest.raises(ValueError):
        KDEClassification("abc")


def test_kde_classifier_fit_invalid(classification_data):
    x_train, labels_train, _ = classification_data
    m_train = x_train.shape[0]
    classifier = KDEClassification()

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        classifier.fit(x_train_tmp, labels_train)

    # Invalid shape of labels_train
    labels_train_tmp = labels_train[:, None]
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train_tmp)

    # Invalid dtype of labels_train
    labels_train_tmp = labels_train * 1.0
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train_tmp)

    # Invalid shape of weights_train
    weights_train = np.ones((m_train, 2))
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train, weights_train)

    # Inconsistent size of x_train and weights_train
    weights_train = np.ones((2 * m_train,))
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train, weights_train)

    # Invalid values of weights_train
    weights_train = np.full((m_train,), -1)
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train, weights_train)

    # Invalid value of bandwidth_method
    with pytest.raises(ValueError):
        classifier.fit(
            x_train,
            labels_train,
            bandwidth_method="abc",
            share_bandwidth=True,
        )

    # Invalid shape of prior_prob
    n_classes = np.unique(labels_train).shape[0]
    prior_prob = np.full((n_classes, 2), 0.5)
    with pytest.raises(ValueError):
        classifier.fit(
            x_train,
            labels_train,
            prior_prob=prior_prob,
        )

    # Invalid size of prior_prob
    n_classes = np.unique(labels_train).shape[0]
    prior_prob = np.full((n_classes + 1,), 0.5)
    with pytest.raises(ValueError):
        classifier.fit(
            x_train,
            labels_train,
            prior_prob=prior_prob,
        )


def test_kde_classifier_predict_invalid(classification_data):
    x_train, labels_train, _ = classification_data
    classifier = KDEClassification()

    # No fitting
    with pytest.raises(RuntimeError):
        classifier.predict(x_train)

    # Invalid shape of x_test
    x_test_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train).predict(x_test_tmp)


def test_kde_classifier_pdfs_invalid(classification_data):
    x_train, labels_train, _ = classification_data
    classifier = KDEClassification()

    # No fitting
    with pytest.raises(RuntimeError):
        classifier.pdfs(x_train)

    # Invalid shape of x_test
    x_test_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        classifier.fit(x_train, labels_train).pdfs(x_test_tmp)
