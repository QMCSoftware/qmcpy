from .copula import (
    AbstractCopula,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_dimension,
)
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2

import numpy as np


def _eulerian_coefficients(n):
    """
    Return Eulerian coefficients for Li_{-n}(z).

    For nonnegative integer n,
    Li_{-n}(z) = z * A_n(z) / (1 - z) ** (n + 1),
    where A_n is the Eulerian polynomial.
    """
    if n == 0:
        return np.array([1.0])

    coefficients = [1]
    for order in range(1, n + 1):
        next_coefficients = []
        for k in range(order):
            left = (k + 1) * coefficients[k] if k < len(coefficients) else 0
            right = (order - k) * coefficients[k - 1] if k > 0 else 0
            next_coefficients.append(left + right)
        coefficients = next_coefficients
    return np.array(coefficients, dtype=float)


class FrankCopula(AbstractCopula):
    r"""
    Frank copula transform with user supplied univariate marginals.

    This implementation supports general dimension for ``theta > 0``. Negative
    ``theta`` is supported only for the bivariate case, where the negative
    parameter Frank copula is valid. For dimensions greater than 2, ``theta``
    must be positive.

    The transform uses the inverse Rosenblatt construction for the Frank
    Archimedean copula. It maps independent uniforms to dependent uniforms by
    recursively inverting conditional CDFs. The base ``AbstractCopula`` class then
    applies each marginal quantile function. SciPy calls the quantile function
    ``ppf``.

    Examples:
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(3, seed=7)
        >>> marginals = [stats.norm(), stats.gamma(a=3), stats.expon()]
        >>> tm = FrankCopula(sampler, marginals=marginals, theta=5.0)
        >>> x = tm(4)
        >>> x.shape
        (4, 3)
        >>> bool(np.isfinite(x).all())
        True
        >>> tm  # doctest: +ELLIPSIS
        FrankCopula (AbstractTrueMeasure)
            marginals       [<...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>]
            theta           5
        >>> rep_tm = FrankCopula(
        ...     DigitalNetB2(3, seed=7, replications=2),
        ...     marginals=marginals,
        ...     theta=5.0,
        ... )
        >>> samples = rep_tm(4)
        >>> samples.shape
        (2, 4, 3)
        >>> bool(np.isfinite(samples).all())
        True
        >>> neg_tm = FrankCopula(DigitalNetB2(2, seed=7), marginals=[stats.uniform(), stats.uniform()], theta=-2.0)
        >>> neg_samples = neg_tm(4)
        >>> neg_samples.shape
        (4, 2)
        >>> bool(((0 <= neg_samples) & (neg_samples <= 1)).all())
        True
        >>> try:
        ...     FrankCopula(DigitalNetB2(3, seed=7), marginals=[stats.uniform()] * 3, theta=-2.0)
        ... except ParameterError as exc:
        ...     print(str(exc))
        theta < 0 is only supported for d=2 FrankCopula.
        >>> FrankCopula(DigitalNetB2(5, seed=7), marginals=[stats.uniform()] * 5, theta=5.0)(4).shape
        (4, 5)

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
            theta (float): Frank dependence parameter. Must be nonzero. Negative
                values are currently supported only for ``d=2``.
        """
        self.parameters = ["marginals", "theta"]
        super(FrankCopula, self).__init__(sampler=sampler, marginals=marginals)
        self.theta = self._parse_theta(theta)

        self._expm1_neg_theta = np.expm1(-self.theta)
        if self._expm1_neg_theta == 0 or not np.isfinite(self._expm1_neg_theta):
            raise ParameterError("theta is too close to 0 or too large in magnitude.")
        self._alpha = -self._expm1_neg_theta
        self._eulerian_cache = {}

    def _parse_theta(self, theta):
        try:
            theta = float(theta)
        except (TypeError, ValueError) as exc:
            raise ParameterError("theta must be a finite nonzero scalar.") from exc

        if not np.isfinite(theta) or theta == 0:
            raise ParameterError("theta must be a finite nonzero scalar.")
        if theta < 0 and self.d != 2:
            raise ParameterError("theta < 0 is only supported for d=2 FrankCopula.")
        return theta

    def _eulerian_coefficients(self, n):
        if n not in self._eulerian_cache:
            self._eulerian_cache[n] = _eulerian_coefficients(n)
        return self._eulerian_cache[n]

    def _q(self, u):
        u = _clip_unit_interval(u)
        q = np.expm1(-self.theta * u) / self._expm1_neg_theta
        eps = np.finfo(float).eps
        return np.clip(q, eps, 1.0 - eps)

    def _z_from_q_product(self, q_product):
        z = self._alpha * q_product
        eps = np.finfo(float).eps
        tiny = np.finfo(float).tiny

        if self.theta > 0:
            return np.clip(z, tiny, 1.0 - eps)
        return np.minimum(z, -tiny)

    def _log_abs_polylog_negative_order(self, z, n):
        # Frank conditional CDFs involve derivatives of the generator that can
        # be very small or very large, so evaluate their absolute value in log space.
        z = np.asarray(z, dtype=float)
        coefficients = self._eulerian_coefficients(n)

        polynomial = np.zeros_like(z, dtype=float)
        for coefficient in coefficients[::-1]:
            polynomial = polynomial * z + coefficient

        tiny = np.finfo(float).tiny
        return (
            np.log(np.maximum(np.abs(z), tiny))
            + np.log(np.maximum(np.abs(polynomial), tiny))
            - (n + 1.0) * np.log1p(-z)
        )

    def _conditional_cdf(self, q_previous, v, derivative_order):
        q_v = self._q(v)
        z_previous = self._z_from_q_product(q_previous)
        z_new = self._z_from_q_product(q_previous * q_v)
        n = derivative_order - 1

        log_conditional = (
            self._log_abs_polylog_negative_order(z_new, n)
            - self._log_abs_polylog_negative_order(z_previous, n)
        )
        return np.clip(np.exp(log_conditional), 0.0, 1.0)

    def _inverse_conditional_cdf(self, q_previous, w, derivative_order):
        w = _clip_unit_interval(w)
        eps = np.finfo(float).eps
        lo = np.full_like(w, eps, dtype=float)
        hi = np.full_like(w, 1.0 - eps, dtype=float)

        for _ in range(60):
            mid = (lo + hi) / 2.0
            conditional_mid = self._conditional_cdf(q_previous, mid, derivative_order)
            lo = np.where(conditional_mid < w, mid, lo)
            hi = np.where(conditional_mid >= w, mid, hi)

        return _clip_unit_interval((lo + hi) / 2.0)

    def _transform_to_uniform(self, x):
        x = _clip_unit_interval(np.asarray(x, dtype=float))
        _validate_dimension(x.shape[-1], self.marginals)

        v = np.empty_like(x, dtype=float)
        v[..., 0] = x[..., 0]
        q_product = self._q(v[..., 0])

        for j in range(1, self.d):
            v[..., j] = self._inverse_conditional_cdf(
                q_product,
                x[..., j],
                derivative_order=j,
            )
            q_product = q_product * self._q(v[..., j])

        return _clip_unit_interval(v)


    def _weight(self, x):
        x = np.asarray(x, dtype=float)
        try:
            u, log_marginal_density = _marginal_cdfs_and_logpdf(x, self.marginals)
        except ParameterError:
            return self._unit_weight_with_warning(x)

        q_product = np.prod(self._q(u), axis=-1)
        z = self._z_from_q_product(q_product)

        log_abs_psi_derivative = (
            self._log_abs_polylog_negative_order(z, self.d - 1)
            - np.log(abs(self.theta))
        )
        log_abs_phi_prime = (
            np.log(abs(self.theta))
            - self.theta * u
            - np.log(np.abs(np.expm1(-self.theta * u)))
        )
        log_copula_density = log_abs_psi_derivative + np.sum(log_abs_phi_prime, axis=-1)

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "FrankCopula can only spawn with the same dimension because "
                "marginals are dimension-specific."
            )
        return FrankCopula(
            sampler=sampler,
            marginals=self.marginals,
            theta=self.theta,
        )
