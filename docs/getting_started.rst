.. _getting_started:

###############
Getting Started
###############

.. currentmodule:: kdelearn


************
Installation
************

Install kdelearn with ``pip``: ::

    $ pip install kdelearn


*************
Example usage
*************

.. plot::
    :align: center
    :include-source:

    from kdelearn.kde import KDE
    from scipy.stats import norm

    # Prepare data
    x_train = np.random.normal(0, 1, (100, 1))
    x_grid = np.linspace(-4, 4, 100).reshape(100, -1)

    # Compute normal distribution on grid (x_test)
    norm_scores = norm.pdf(x_grid)

    # Compute kernel density estimation on grid (x_grid)
    kde = KDE().fit(x_train)
    kde_scores = kde.pdf(x_test)

    plt.plot(x_grid, norm_scores, label="normal distribution")
    plt.plot(x_grid, kde_scores, label="kde")
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(0, 0.45)
    plt.xlabel("$x$", fontsize=11)
    plt.grid(linestyle="--")
    plt.show()
