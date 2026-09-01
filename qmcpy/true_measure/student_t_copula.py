from .copula import (
    AbstractCopula,
    _clip_unit_interval,
    _marginal_cdfs_and_logpdf,
    _validate_correlation_matrix,
    _validate_dimension,
)
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2

import numpy as np
import scipy.stats as stats


class StudentTCopula(AbstractCopula):
    r"""
    Student-t copula transform with user supplied univariate marginals.

    This TrueMeasure uses the same marginal workflow as ``GaussianCopula``,
    but builds dependent uniforms through a multivariate Student-t copula with
    correlation matrix ``correlation`` and degrees of freedom ``df``.

    The transform uses the inverse Rosenblatt construction for the
    multivariate Student-t distribution. This is equivalent in distribution to
    the standard correlated-normal plus shared chi-square scaling construction,
    but it only needs d deterministic uniforms from the base QMCPy sampler.
    It is not the incorrect shortcut of applying univariate ``t.ppf``, a
    Cholesky factor, and then univariate ``t.cdf``.

    Examples:
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> sampler = DigitalNetB2(2, seed=7)
        >>> marginals = [stats.norm(), stats.gamma(a=3, scale=2)]
        >>> corr = [[1.0, 0.6], [0.6, 1.0]]
        >>> tm = StudentTCopula(sampler, marginals=marginals, correlation=corr, df=4)
        >>> x = tm(4)
        >>> x.shape
        (4, 2)
        >>> bool(np.isfinite(x).all())
        True
        >>> tm  # doctest: +ELLIPSIS
        StudentTCopula (AbstractTrueMeasure)
            marginals       [<...rv_continuous_frozen object at ...>
                             <...rv_continuous_frozen object at ...>]
            correlation     [[1.  0.6]
                             [0.6 1. ]]
            df              2^(2)
        >>> rep_marginals = [stats.norm(), stats.gamma(a=3, scale=2), stats.expon()]
        >>> rep_corr = [[1.0, 0.6, 0.3],
        ...             [0.6, 1.0, 0.2],
        ...             [0.3, 0.2, 1.0]]
        >>> rep_tm = StudentTCopula(
        ...     DigitalNetB2(3, seed=7, replications=2),
        ...     marginals=rep_marginals,
        ...     correlation=rep_corr,
        ...     df=4,
        ... )
        >>> samples = rep_tm(4)
        >>> samples.shape
        (2, 4, 3)
        >>> bool(np.isfinite(samples).all())
        True
        >>> StudentTCopula(DigitalNetB2(1, seed=7), marginals=[stats.norm()], correlation=[[1.0]], df=4)(4).shape
        (4, 1)
        >>> StudentTCopula(DigitalNetB2(2, seed=7), marginals=marginals, correlation=corr, df=1)(4).shape
        (4, 2)

    **References:**

    1.  Roger B. Nelsen. *An Introduction to Copulas*. Second Edition,
        Springer Series in Statistics, Springer, 2006.
        [doi:10.1007/0-387-28678-0](https://doi.org/10.1007/0-387-28678-0).

    2.  Mathieu Cambou, Marius Hofert, and Christiane Lemieux.
        "Quasi-random numbers for copula models."
        [arXiv:1508.03483](https://arxiv.org/abs/1508.03483).

    3.  M. Rosenblatt. "Remarks on a Multivariate Transformation."
        The Annals of Mathematical Statistics 23(3), 470-472, 1952.
        [doi:10.1214/aoms/1177729394](https://doi.org/10.1214/aoms/1177729394).
    """

    _missing_weight_warning_message = (
        "StudentTCopula needs scipy.stats.multivariate_t and marginals with "
        "'cdf' and 'pdf' or 'logpdf' to compute density weights. "
        "Weights will be treated as 1."
    )

    def __init__(self, sampler, marginals, correlation, df):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A sampler or transform whose range is the unit cube.
            marginals (list): Length d list of SciPy-like univariate
                distributions implementing a quantile function, called ``ppf``
                in SciPy.
            correlation (np.ndarray): d x d positive definite correlation matrix.
            df (float): Positive Student-t degrees of freedom.
        """
        self.parameters = ["marginals", "correlation", "df"]
        super(StudentTCopula, self).__init__(sampler=sampler, marginals=marginals)
        self.correlation = _validate_correlation_matrix(correlation, self.d)
        self.df = self._parse_df(df)

        self._mvt_scipy = None
        if hasattr(stats, "multivariate_t"):
            self._mvt_scipy = stats.multivariate_t(
                loc=np.zeros(self.d), shape=self.correlation, df=self.df
            )

    def _parse_df(self, df):
        try:
            df = float(df)
        except (TypeError, ValueError) as exc:
            raise ParameterError("df must be a positive scalar.") from exc

        if not np.isfinite(df) or df <= 0:
            raise ParameterError("df must be a positive scalar.")
        return df

    def _dependent_t_samples(self, u):
        """
        Map independent uniforms to a multivariate Student-t sample.

        A direct scale-mixture construction would need d normal uniforms plus
        one extra chi-square uniform for the shared radial scale. Since
        TrueMeasure transforms are dimension preserving, we instead use the
        exact conditional Student-t inverse CDFs. This is the inverse
        Rosenblatt transform of the same multivariate Student-t law and
        preserves the shared-tail dependence of the t copula.
        """
        u = _clip_unit_interval(np.asarray(u, dtype=float))
        _validate_dimension(u.shape[-1], self.marginals)

        orig_shape = u.shape[:-1]
        uu = u.reshape(-1, self.d)
        z = np.empty_like(uu, dtype=float)

        z[:, 0] = stats.t.ppf(uu[:, 0], df=self.df)

        for i in range(1, self.d):
            A = slice(0, i)

            corr_AA = self.correlation[A, A]
            corr_BA = self.correlation[i, A]
            corr_AB = self.correlation[A, i]
            corr_BB = self.correlation[i, i]

            z_A = z[:, A]
            sol = np.linalg.solve(corr_AA, z_A.T).T
            d_A = np.sum(z_A * sol, axis=1)

            loc_cond = sol @ corr_BA

            corr_AA_inv_corr_AB = np.linalg.solve(corr_AA, corr_AB)
            schur = corr_BB - corr_BA @ corr_AA_inv_corr_AB

            df_cond = self.df + i
            shape_cond = (self.df + d_A) / (self.df + i) * schur
            shape_cond = np.maximum(shape_cond, np.finfo(float).tiny)

            z[:, i] = stats.t.ppf(
                uu[:, i],
                df=df_cond,
                loc=loc_cond,
                scale=np.sqrt(shape_cond),
            )

        return z.reshape(*orig_shape, self.d)

    def _transform_to_uniform(self, x):
        z = self._dependent_t_samples(x)
        return _clip_unit_interval(stats.t.cdf(z, df=self.df))


    def _weight(self, x):
        x = np.asarray(x, dtype=float)

        if self._mvt_scipy is None:
            return self._unit_weight_with_warning(x)

        try:
            u, log_marginal_density = _marginal_cdfs_and_logpdf(x, self.marginals)
        except ParameterError:
            return self._unit_weight_with_warning(x)

        z = stats.t.ppf(u, df=self.df)
        z_flat = z.reshape(-1, self.d)

        log_joint = self._mvt_scipy.logpdf(z_flat).reshape(z.shape[:-1])
        log_independent = np.sum(stats.t.logpdf(z, df=self.df), axis=-1)
        log_copula_density = log_joint - log_independent

        return np.exp(log_copula_density + log_marginal_density)

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "StudentTCopula can only spawn with the same dimension because "
                "marginals and correlation are dimension-specific."
            )
        return StudentTCopula(
            sampler=sampler,
            marginals=self.marginals,
            correlation=self.correlation,
            df=self.df,
        )
