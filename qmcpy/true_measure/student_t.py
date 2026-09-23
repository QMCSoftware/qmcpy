import numpy as np
import scipy.stats as stats

from ..util import ParameterError, DimensionError
from .scipy_wrapper import SciPyWrapper


class _StudentTAdapter:
    """Multivariate Student t adapter for SciPyWrapper.

    - transform(u): sequential conditioning using univariate t conditionals
    - logpdf(x): forwarded to scipy.stats.multivariate_t (if available)
    """

    def __init__(self, loc, shape, df):
        self.loc = np.asarray(loc, dtype=float)
        self.shape = np.asarray(shape, dtype=float)
        self.df = float(df)

        if self.loc.ndim != 1:
            raise ParameterError("loc must be a 1D vector.")
        if self.shape.ndim != 2 or self.shape.shape[0] != self.shape.shape[1]:
            raise ParameterError("shape must be a square matrix.")
        if self.shape.shape[0] != self.loc.size:
            raise DimensionError("loc and shape dimensions do not match.")
        if self.df <= 0:
            raise ParameterError("df must be positive.")

        # SciPy compatibility guard
        if not hasattr(stats, "multivariate_t"):
            raise ImportError(
                "scipy.stats.multivariate_t is not available in the installed SciPy. "
                "StudentT requires a SciPy version that provides scipy.stats.multivariate_t."
            )

        self.dim = self.loc.size
        self._rv = stats.multivariate_t(loc=self.loc, shape=self.shape, df=self.df)

    @staticmethod
    def _clip_u(u):
        eps = np.finfo(float).eps
        return np.clip(u, eps, 1.0 - eps)

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != self.dim:
            raise DimensionError(f"Expected last axis {self.dim}, got {u.shape[-1]}")

        u = self._clip_u(u)

        orig_shape = u.shape[:-1]
        n = int(np.prod(orig_shape)) if orig_shape else 1
        uu = u.reshape(n, self.dim)

        x = np.empty_like(uu)

        scale0 = np.sqrt(self.shape[0, 0])
        x[:, 0] = stats.t.ppf(uu[:, 0], df=self.df, loc=self.loc[0], scale=scale0)

        for i in range(1, self.dim):
            A = slice(0, i)

            mu_A = self.loc[A]
            mu_B = self.loc[i]

            Sigma_AA = self.shape[A, A]
            Sigma_BA = self.shape[i, A]
            Sigma_AB = self.shape[A, i]
            Sigma_BB = self.shape[i, i]

            x_A = x[:, A]
            diff = x_A - mu_A

            sol = np.linalg.solve(Sigma_AA, diff.T).T
            d_A = np.sum(diff * sol, axis=1)

            mu_cond = mu_B + sol @ Sigma_BA

            Sigma_AA_inv_Sigma_AB = np.linalg.solve(Sigma_AA, Sigma_AB)
            schur = Sigma_BB - Sigma_BA @ Sigma_AA_inv_Sigma_AB

            df_cond = self.df + i
            shape_cond = (self.df + d_A) / (self.df + i) * schur
            shape_cond = np.maximum(shape_cond, np.finfo(float).tiny)

            x[:, i] = stats.t.ppf(
                uu[:, i],
                df=df_cond,
                loc=mu_cond,
                scale=np.sqrt(shape_cond),
            )

        return x.reshape(*orig_shape, self.dim)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.dim:
            raise DimensionError(f"Expected last axis {self.dim}, got {x.shape[-1]}")
        x_flat = x.reshape(-1, self.dim)
        lp = self._rv.logpdf(x_flat)
        return np.asarray(lp).reshape(x.shape[:-1])


class StudentT(SciPyWrapper):
    """Convenience true measure: multivariate Student t.
    """

    def __init__(self, sampler, loc, shape, df) -> None:
        super().__init__(
            sampler=sampler,
            scipy_distribs=_StudentTAdapter(loc=loc, shape=shape, df=df),
        )
