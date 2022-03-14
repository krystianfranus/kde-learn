Kernels
=======

Available kernel functions :math:`K(x)`

Formulas
--------

.. table:: Formulas for available kernel functions
    :widths: auto

    ==============  =============================================================================
    Kernel name     Formula
    ==============  =============================================================================
    Gaussian        :math:`\frac{1}{\sqrt{2 \pi}} \exp \left( \frac{x^2}{2} \right)`
    Uniform         :math:`0.5 \quad \text{if } |x| \leq 0 \quad \text{else } 0`
    Epanechnikov    :math:`\frac{3}{4} (1-x^2) \quad \text{if } |x| \leq 0 \quad \text{else } 0`
    Cauchy          :math:`\frac{2}{\pi (x^2 + 1)^2}`
    ==============  =============================================================================

Comparison plot
---------------

.. plot::
    :include-source:

    from kdelearn.cutils import gaussian, uniform, epanechnikov, cauchy
    m_test = 1000
    x_test = np.linspace(-4, 4, m_test)

    plt.figure(figsize=(6, 4))
    for kernel in [gaussian, uniform, epanechnikov, cauchy]:
        scores = [kernel(x_test[i]) for i in range(m_test)]
        plt.plot(x_test, scores, label=kernel.__name__)
    plt.ylim(top=0.8); plt.legend(); plt.grid();
