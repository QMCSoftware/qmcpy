from .copula import (
    AbstractCopula,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_dimension,
)
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2

import numpy as np


class GumbelCopula(AbstractCopula):
    r"""
    Gumbel copula transform with user supplied marginals.

    This implementation supports general dimension for ``theta >= 1``. It
    maps independent uniforms to Gumbel-dependent uniforms by numerically
    inverting the conditional CDFs from the inverse Rosenblatt construction.
    The base ``AbstractCopula`` class then applies marginal quantile functions.
    SciPy calls the quantile function ``ppf``.

    Gumbel copulas have positive upper-tail dependence for ``theta > 1``.
    The boundary case ``theta = 1`` is the independent copula.

    Examples:
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(2, seed=7)
        >>> marginals = [stats.expon(), stats.gamma(a=3)]
        >>> tm = GumbelCopula(sampler, marginals=marginals, theta=2.0)
        >>> x = tm(4)
        >>> x.shape
        (4, 2)
        >>> bool(np.isfinite(x).all())
        True
        >>> tm  # doctest: +ELLIPSIS
        GumbelCopula (AbstractTrueMeasure)
            marginals       [<...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>]
            theta           2^(1)
        >>> rep_marginals = [stats.expon(), stats.gamma(a=3), stats.beta(a=2, b=5)]
        >>> rep_tm = GumbelCopula(
        ...     DigitalNetB2(3, seed=7, replications=2),
        ...     marginals=rep_marginals,
        ...     theta=2.0,
        ... )
        >>> samples = rep_tm(4)
        >>> samples.shape
        (2, 4, 3)
        >>> bool(np.isfinite(samples).all())
        True
        >>> GumbelCopula(DigitalNetB2(3, seed=7), marginals=[stats.uniform()] * 3, theta=2.0)(4).shape
        (4, 3)
        >>> independent_tm = GumbelCopula(DigitalNetB2(2, seed=7), marginals=[stats.uniform(), stats.uniform()], theta=1.0)
        >>> independent_samples = independent_tm(4)
        >>> independent_samples.shape
        (4, 2)
        >>> bool(((0 <= independent_samples) & (independent_samples <= 1)).all())
        True

    **References:**

    1.  Roger B. Nelsen. *An Introduction to Copulas*. Second Edition,
        Springer Series in Statistics, Springer, 2006.
        [doi:10.1007/0-387-28678-0](https://doi.org/10.1007/0-387-28678-0).

    2.  Mathieu Cambou, Marius Hofert, and Christiane Lemieux.
        "Quasi-random numbers for copula models."
        [arXiv:1508.03483](https://arxiv.org/abs/1508.03483).

    3.  Marius Hofert, Martin Maechler, and Alexander J. McNeil.
        "Likelihood inference for Archimedean copulas in high dimensions
        under known margins." Journal of Multivariate Analysis 110,
        133-150, 2012.
        [doi:10.1016/j.jmva.2012.02.019](https://doi.org/10.1016/j.jmva.2012.02.019).
    """

    def __init__(self, sampler, marginals, theta):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A sampler or transform whose range is the unit cube.
            marginals (list): Length d list of SciPy-like univariate
                distributions implementing a quantile function, called ``ppf``
                in SciPy.
            theta (float): Gumbel dependence parameter, requiring ``theta >= 1``.
        """
        self.parameters = ["marginals", "theta"]
        super(GumbelCopula, self).__init__(sampler=sampler, marginals=marginals)
        self.theta = self._parse_theta(theta)
        self._alpha = 1.0 / self.theta
        self._derivative_terms_cache = {}

    def _parse_theta(self, theta):
        try:
            theta = float(theta)
        except (TypeError, ValueError) as exc:
            raise ParameterError("theta must be a scalar greater than or equal to 1.") from exc

        if not np.isfinite(theta) or theta < 1:
            raise ParameterError("theta must be a scalar greater than or equal to 1.")
        return theta

    def _phi(self, u):
        u = _clip_unit_interval(u)
        return (-np.log(u)) ** self.theta

    def _derivative_terms(self, order):
        if order not in self._derivative_terms_cache:
            terms = [(1.0, 0.0)]
            for _ in range(order):
                next_terms = []
                for coefficient, exponent in terms:
                    next_terms.append(
                        (coefficient * self._alpha, exponent + self._alpha - 1.0)
                    )
                    if exponent != 0.0:
                        next_terms.append((-coefficient * exponent, exponent - 1.0))
                terms = next_terms
            self._derivative_terms_cache[order] = terms
        return self._derivative_terms_cache[order]

    def _log_polynomial(self, t, order):
        t = np.maximum(np.asarray(t, dtype=float), np.finfo(float).tiny)
        log_t = np.log(t)
        logs = []
        for coefficient, exponent in self._derivative_terms(order):
            logs.append(np.log(coefficient) + exponent * log_t)
        return np.logaddexp.reduce(np.stack(logs, axis=0), axis=0)

    def _log_abs_psi_derivative(self, t, order):
        # Conditional CDFs and densities use ratios of generator derivatives.
        # Taking logs keeps those ratios stable for small uniforms and large theta.
        t = np.maximum(np.asarray(t, dtype=float), np.finfo(float).tiny)
        return -(t ** self._alpha) + self._log_polynomial(t, order)

    def _conditional_cdf(self, s_previous, v, derivative_order):
        v = _clip_unit_interval(v)

        if self.theta == 1.0:
            return v

        s_previous = np.maximum(np.asarray(s_previous, dtype=float), np.finfo(float).tiny)
        s_new = s_previous + self._phi(v)
        log_conditional = (
            self._log_abs_psi_derivative(s_new, derivative_order)
            - self._log_abs_psi_derivative(s_previous, derivative_order)
        )
        return np.clip(np.exp(log_conditional), 0.0, 1.0)

    def _inverse_conditional_cdf(self, s_previous, w, derivative_order):
        w = _clip_unit_interval(w)

        if self.theta == 1.0:
            return w

        eps = np.finfo(float).eps
        lo = np.full_like(w, eps, dtype=float)
        hi = np.full_like(w, 1.0 - eps, dtype=float)

        for _ in range(60):
            mid = (lo + hi) / 2.0
            cond_mid = self._conditional_cdf(s_previous, mid, derivative_order)
            lo = np.where(cond_mid < w, mid, lo)
            hi = np.where(cond_mid >= w, mid, hi)

        return _clip_unit_interval((lo + hi) / 2.0)

    def _transform_to_uniform(self, x):
        x = _clip_unit_interval(np.asarray(x, dtype=float))
        _validate_dimension(x.shape[-1], self.marginals)

        if self.theta == 1.0:
            return x

        v = np.empty_like(x, dtype=float)
        v[..., 0] = x[..., 0]
        s_previous = self._phi(v[..., 0])

        for j in range(1, self.d):
            v[..., j] = self._inverse_conditional_cdf(
                s_previous,
                x[..., j],
                derivative_order=j,
            )
            s_previous = s_previous + self._phi(v[..., j])

        return _clip_unit_interval(v)


    def _weight(self, x):
        x = np.asarray(x, dtype=float)
        try:
            u, log_marginal_density = _marginal_cdfs_and_logpdf(x, self.marginals)
        except ParameterError:
            return self._unit_weight_with_warning(x)

        u = _clip_unit_interval(u)

        if self.theta == 1.0:
            return np.exp(log_marginal_density)

        log_u = np.log(u)
        log_neg_log_u = np.log(-log_u)
        s = np.sum(self._phi(u), axis=-1)
        log_abs_phi_prime = (
            np.log(self.theta)
            + (self.theta - 1.0) * log_neg_log_u
            - log_u
        )
        log_copula_density = (
            self._log_abs_psi_derivative(s, self.d)
            + np.sum(log_abs_phi_prime, axis=-1)
        )

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "GumbelCopula can only spawn with the same dimension because "
                "marginals are dimension-specific."
            )
        return GumbelCopula(
            sampler=sampler,
            marginals=self.marginals,
            theta=self.theta,
        )
