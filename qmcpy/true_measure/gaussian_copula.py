from .copula import (
    AbstractCopula,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_correlation_matrix,
    _validate_dimension,
)
from .gaussian import Gaussian
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2

import numpy as np
from scipy.stats import norm


class GaussianCopula(AbstractCopula):
    r"""
    Gaussian copula transform with user supplied univariate marginals.

    This TrueMeasure separates the dependence model from the marginal
    distributions:

    1. map independent uniforms through ``scipy.stats.norm.ppf``;
    2. inject Gaussian dependence with a Cholesky factor of the correlation;
    3. map back to dependent uniforms with ``scipy.stats.norm.cdf``;
    4. apply each marginal quantile function.

    SciPy calls the quantile function ``ppf``. The marginal objects must expose
    this method. If they also expose
    ``cdf`` and ``pdf`` or ``logpdf``, then ``_weight`` computes the Gaussian
    copula joint density. Otherwise weights are treated as one with a warning.

    Examples:
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(2, seed=7)
        >>> marginals = [stats.beta(a=2, b=5), stats.gamma(a=3, scale=2)]
        >>> corr = [[1.0, 0.6], [0.6, 1.0]]
        >>> tm = GaussianCopula(sampler, marginals=marginals, correlation=corr)
        >>> x = tm(4)
        >>> x.shape
        (4, 2)
        >>> bool(np.isfinite(x).all())
        True
        >>> tm  # doctest: +ELLIPSIS
        GaussianCopula (AbstractTrueMeasure)
            marginals       [<...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>]
            correlation     [[1.  0.6]
                             [0.6 1. ]]
        >>> rep_marginals = [stats.beta(a=2, b=5), stats.gamma(a=3, scale=2), stats.expon()]
        >>> rep_corr = [[1.0, 0.6, 0.3],
        ...             [0.6, 1.0, 0.2],
        ...             [0.3, 0.2, 1.0]]
        >>> rep_tm = GaussianCopula(
        ...     DigitalNetB2(3, seed=7, replications=2),
        ...     marginals=rep_marginals,
        ...     correlation=rep_corr,
        ... )
        >>> samples = rep_tm(4)
        >>> samples.shape
        (2, 4, 3)
        >>> bool(np.isfinite(samples).all())
        True
        >>> GaussianCopula(DigitalNetB2(1, seed=7), marginals=[stats.norm()], correlation=[[1.0]])(4).shape
        (4, 1)

    **References:**

    1.  Roger B. Nelsen. *An Introduction to Copulas*. Second Edition,
        Springer Series in Statistics, Springer, 2006.
        [doi:10.1007/0-387-28678-0](https://doi.org/10.1007/0-387-28678-0).

    2.  Mathieu Cambou, Marius Hofert, and Christiane Lemieux.
        "Quasi-random numbers for copula models."
        [arXiv:1508.03483](https://arxiv.org/abs/1508.03483).
    """

    def __init__(self, sampler, marginals, correlation):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A sampler or transform whose range is the unit cube.
            marginals (list): Length d list of SciPy-like univariate
                distributions implementing a quantile function, called ``ppf``
                in SciPy.
            correlation (np.ndarray): d x d positive definite correlation matrix.
        """
        self.parameters = ["marginals", "correlation"]
        super(GaussianCopula, self).__init__(sampler=sampler, marginals=marginals)
        self.correlation = _validate_correlation_matrix(correlation, self.d)

        self._gaussian_transform = Gaussian(
            sampler,
            mean=np.zeros(self.d),
            covariance=self.correlation,
            decomp_type="Cholesky",
        )
        # Gaussian copula density:
        # c(u) = |R|^{-1/2} exp(-0.5 z^T (R^{-1} - I) z), z = Phi^{-1}(u).
        # The identity subtraction removes the independent standard-normal density
        # already accounted for by the marginal normal transforms.
        self._corr_inv_minus_eye = np.linalg.inv(self.correlation) - np.eye(self.d)
        _, self._logdet_corr = np.linalg.slogdet(self.correlation)


    def _transform_to_uniform(self, x):
        x = np.asarray(x, dtype=float)
        _validate_dimension(x.shape[-1], self.marginals)

        u = _clip_unit_interval(x)
        z_dep = self._gaussian_transform._transform(u)
        return _clip_unit_interval(norm.cdf(z_dep))


    def _weight(self, x):
        x = np.asarray(x, dtype=float)
        try:
            u, log_marginal_density = _marginal_cdfs_and_logpdf(x, self.marginals)
        except ParameterError:
            return self._unit_weight_with_warning(x)

        z = norm.ppf(u)
        quad = np.einsum("...i,ij,...j->...", z, self._corr_inv_minus_eye, z)
        log_copula_density = -0.5 * self._logdet_corr - 0.5 * quad

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "GaussianCopula can only spawn with the same dimension because "
                "marginals and correlation are dimension-specific."
            )
        return GaussianCopula(
            sampler=sampler,
            marginals=self.marginals,
            correlation=self.correlation,
        )
