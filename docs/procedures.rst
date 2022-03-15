Procedures
==========

.. currentmodule:: kdelearn

kde classifier
--------------

Classifier description

.. math::
    P(C=c|X=x) \propto \alpha_c \hat{f}_c(X=x)
.. math::
    \underset{c}{\mathrm{argmax}} \quad P(C=c|X=x)

Example:

.. plot::
    :include-source:

    from kdelearn.kde_funcs import kde_classifier

    # Prepare train data
    m_train = 1000
    cov = [[1, 0], [0, 1]]

    ## class 1
    x_train1 = np.random.multivariate_normal([0, 0], cov, m_train // 2)
    labels_train1 = np.full(m_train // 2, 1)
    ## class 2
    x_train2 = np.random.multivariate_normal([3, 3], cov, m_train // 2)
    labels_train2 = np.full(m_train // 2, 2)

    x_train = np.concatenate((x_train1, x_train2))  # shape (1000, 2)
    labels_train = np.concatenate((labels_train1, labels_train2))  # shape (1000,)

    # Prepare 2d grid for plotting decision boundary
    grid_size = 70
    x1 = np.linspace(-6, 8, grid_size)
    x2 = np.linspace(-6, 8, grid_size)
    x1v, x2v = np.meshgrid(x1, x2)
    x1p = x1v.reshape(-1, 1)
    x2p = x2v.reshape(-1, 1)
    x_test = np.hstack((x1p, x2p))  # shape (10000, 2)

    # Classify grid points
    labels_pred = kde_classifier(x_train, labels_train, x_test)

    for label, color in zip(np.unique(labels_train), ["cornflowerblue", "goldenrod"]):
        mask1 = (labels_train == label)
        plt.scatter(x_train[mask1, 0], x_train[mask1, 1], facecolors="none",
                    edgecolors=color, label=f"class {label}")
        mask2 = (labels_pred == label)
        plt.scatter(x1p[mask2, 0], x2p[mask2, 0],
                    color=color, marker=".", alpha=0.15)
    plt.xlim(-6, 8); plt.ylim(-6, 8);
    plt.xlabel("$x_1$", fontsize=11); plt.ylabel("$x_2$", rotation=0, fontsize=11);
    plt.legend(); plt.title("Decision boundry determined by kde classifier", fontsize=11);
