.. kde-learn documentation master file, created by
   sphinx-quickstart on Mon Apr 12 23:15:08 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


#####################################
Welcome to kde-learn's documentation!
#####################################

.. currentmodule:: kdelearn

*Kdelearn* is a python library that gives you the ability to solve three fundamental
tasks in data analysis:

- classification,
- outliers detection,
- clustering.

All the procedures are based on kernel density estimation (non-parametric density
estimation method) both in unconditional (standard) and conditional case.


************
Installation
************

Install *kdelearn* with ``pip``: ::

    $ pip install kdelearn


*************
Example usage
*************

.. plot::
    :align: center
    :include-source:

    import numpy as np
    from matplotlib import pyplot as plt
    from kdelearn.kde import KDE
    from scipy.stats import norm

    np.random.seed(0)

    # Prepare data
    x_train = np.random.normal(0, 1, (100, 1))
    x_grid = np.linspace(-4, 4, 1000).reshape(1000, -1)

    # Compute normal distribution on grid (x_grid)
    norm_scores = norm.pdf(x_grid)

    # Compute kernel density estimation on grid (x_grid)
    kde = KDE().fit(x_train)
    kde_scores = kde.pdf(x_grid)

    # Plot
    plt.plot(x_grid, norm_scores, label="normal distribution")
    plt.plot(x_grid, kde_scores, label="kde")

    plt.legend(fontsize=10)
    plt.xlim(-4, 4)
    plt.ylim(0, 0.45)
    plt.xlabel("$x$", fontsize=11)
    plt.grid(linestyle="--")
    plt.show()

.. toctree::
   :caption: Contents
   :maxdepth: 2

   density_estimation_desc.rst
   tasks.rst

.. toctree::
   :caption: Api Reference
   :maxdepth: 2

   unconditional_api.rst
   conditional_api.rst
   bandwidth_selection.rst
   metrics.rst


******************
Indices and tables
******************

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
