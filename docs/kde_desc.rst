Kernel density estimation
=========================

Formula
-------

.. math::
    \hat{f}(x) = \sum_{i=1}^m w_{i} \prod_{j=i}^n \frac{1}{h_j} K \left( \frac{x_{j} - x_{i, j}}{h_j} \right) \text{,} \quad x \in \mathbb{R}^n

.. hlist::
    :columns: 2

    - :math:`m` - size of dataset
    - :math:`w` - weights of dataset
    - :math:`h` - bandwidth (smoothing parameter)
    - :math:`n` - dimensionality
    - :math:`K(x)` - kernel function


Kernels
-------

.. table:: Formulas for available kernel functions
    :widths: auto

    ==============  =============================================================================
    Kernel name     Formula
    ==============  =============================================================================
    Gaussian        :math:`\frac{1}{\sqrt{2 \pi}} \exp \left( \frac{x^2}{2} \right)`
    Uniform         :math:`0.5 \quad \text{if } |x| \leq 0 \quad \text{otherwise } 0`
    Epanechnikov    :math:`\frac{3}{4} (1-x^2) \quad \text{if } |x| \leq 0 \quad \text{otherwise } 0`
    Cauchy          :math:`\frac{2}{\pi (x^2 + 1)^2}`
    ==============  =============================================================================

Comparison plot

.. plot::
    :include-source:

    from kdelearn.cutils import gaussian, uniform, epanechnikov, cauchy

    m_test = 1000
    x_test = np.linspace(-3, 3, m_test)

    for kernel in [gaussian, uniform, epanechnikov, cauchy]:
        scores = [kernel(x_test[i]) for i in range(m_test)]
        plt.plot(x_test, scores, label=kernel.__name__)
    plt.ylim(top=0.8); plt.legend(); plt.grid();
    plt.xlabel("$x$", fontsize=11); plt.ylabel("$K(x)$", rotation=0, labelpad=15, fontsize=11);
    plt.title("Plot of available kernel functions", fontsize=11)
