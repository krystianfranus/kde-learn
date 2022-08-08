import numpy as np
import pytest

from kdelearn.kde_funcs import KDEClassifier


@pytest.mark.parametrize(
    "kernel_name", ["gaussian", "uniform", "epanechnikov", "cauchy"]
)
@pytest.mark.parametrize("share_bandwidth", [False, True])
@pytest.mark.parametrize(
    "bandwidth_method", ["normal_reference", "direct_plugin", "ste_plugin", "ml_cv"]
)
def test_kde_classifier(
    data_classification,
    kernel_name,
    share_bandwidth,
    bandwidth_method,
):
    x_train, labels_train, x_test, labels_test = data_classification
    kde_classifier = KDEClassifier(kernel_name)
    kde_classifier = kde_classifier.fit(x_train, labels_train)
    labels_pred = kde_classifier.predict(x_test)
    assert kde_classifier.fitted
    assert labels_pred.shape[0] == x_test.shape[0]
    assert labels_pred.shape[0] == labels_test.shape[0]
    assert labels_pred.ndim == labels_test.ndim


def test_kde_classifier_with_weights_train(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    m_train = x_train.shape[0]
    weights_train = np.ones((m_train,))
    kde_classifier = KDEClassifier()
    kde_classifier.fit(x_train, labels_train, weights_train=weights_train)
    assert kde_classifier.fitted


def test_kde_with_fixed_bandwidth(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    n = x_train.shape[1]
    bandwidth = np.array([0.5] * n)
    kde_classifier = KDEClassifier()
    kde_classifier.fit(x_train, labels_train, bandwidth=bandwidth)
    assert kde_classifier.fitted


def test_kde_with_fixed_prior_prob(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    ulabels = np.unique(labels_train)
    n_classes = ulabels.shape[0]
    prior_prob = np.array([0.5] * n_classes)
    kde_classifier = KDEClassifier()
    kde_classifier.fit(x_train, labels_train, prior_prob=prior_prob)
    assert kde_classifier.fitted


def test_kde_classifier_pdfs(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    ulabels = np.unique(labels_train)
    n_classes = ulabels.shape[0]
    kde_classifier = KDEClassifier()
    scores = kde_classifier.fit(x_train, labels_train).pdfs(x_test)
    assert kde_classifier.fitted
    assert scores.ndim == 2
    assert scores.shape[1] == n_classes
    assert scores.shape[0] == x_test.shape[0]
    assert (scores >= 0).all()
    assert (scores < 1).all()


def test_kde_classifier_with_invalid_kernel_name():
    with pytest.raises(ValueError):
        KDEClassifier("abc")


def test_kde_classifier_fit_with_invalid_data(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    m_train = x_train.shape[0]
    kde_classifier = KDEClassifier()

    # Invalid shape of x_train
    x_train_tmp = x_train.flatten()
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train_tmp, labels_train)

    # Invalid shape of labels_train
    labels_train_tmp = labels_train[:, None]
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train, labels_train_tmp)

    # Invalid dtype of labels_train
    labels_train_tmp = labels_train * 1.0
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train, labels_train_tmp)

    # Invalid shape of weights_train
    weights_train = np.ones((m_train, 2))
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train, labels_train, weights_train)

    # Invalid values of weights_train
    weights_train = np.full((m_train,), -1)
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train, labels_train, weights_train)

    # Invalid value of bandwidth_method
    with pytest.raises(ValueError):
        kde_classifier.fit(
            x_train,
            labels_train,
            share_bandwidth=True,
            bandwidth_method="abc",
        )

    # Invalid shape of prior_prob
    n_classes = np.unique(labels_train).shape[0]
    prior_prob = np.full((n_classes, 2), 0.5)
    with pytest.raises(ValueError):
        kde_classifier.fit(
            x_train,
            labels_train,
            prior_prob=prior_prob,
        )

    # Invalid number of classes in prior_prob
    n_classes = np.unique(labels_train).shape[0]
    prior_prob = np.full((n_classes + 1,), 0.5)
    with pytest.raises(ValueError):
        kde_classifier.fit(
            x_train,
            labels_train,
            prior_prob=prior_prob,
        )


def test_kde_classifier_predict_invalid_data(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    kde_classifier = KDEClassifier()

    # Invalid shape of x_test
    x_test_tmp = x_test.flatten()
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train, labels_train).predict(x_test_tmp)


def test_kde_classifier_pdfs_invalid_data(data_classification):
    x_train, labels_train, x_test, labels_test = data_classification
    kde_classifier = KDEClassifier()

    # Invalid shape of x_test
    x_test_tmp = x_test.flatten()
    with pytest.raises(ValueError):
        kde_classifier.fit(x_train, labels_train).pdfs(x_test_tmp)
