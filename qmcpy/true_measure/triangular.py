import numpy as np

from ..util import ParameterError
from .scipy_wrapper import SciPyWrapper


class TriangularDistribution:
    """Triangular distribution matching scipy.stats.triang behavior.

    Support: [loc, loc + scale] Mode: loc + c*scale, with 0 < c < 1 Provides
    ppf and pdf for SciPyWrapper custom-marginal usage.
    """

    def __init__(self, c=0.5, loc=0.0, scale=1.0) -> None:
        c = float(c)
        loc = float(loc)
        scale = float(scale)

        if not (0.0 < c < 1.0):
            raise ParameterError("c must lie strictly between 0 and 1.")
        if scale <= 0.0:
            raise ParameterError("scale must be positive.")

        self.c = c
        self.loc = loc
        self.scale = scale

        self._a = loc
        self._b = loc + scale
        self._m = loc + c * scale

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function of the triangular distribution.

        Args:
            x (np.ndarray): Points at which to evaluate the density.

        Returns:
            np.ndarray: Density values, same shape as `x`.
        """
        x = np.asarray(x, dtype=float)
        a, m, b = self._a, self._m, self._b
        out = np.zeros_like(x, dtype=float)

        left = (x >= a) & (x < m)
        right = (x >= m) & (x <= b)

        out[left] = 2.0 * (x[left] - a) / ((b - a) * (m - a))
        out[right] = 2.0 * (b - x[right]) / ((b - a) * (b - m))
        return out

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF) of the triangular distribution.

        Args:
            u (np.ndarray): Probabilities in `[0,1]` at which to evaluate the inverse CDF.

        Returns:
            np.ndarray: Quantile values, same shape as `u`.
        """
        u = np.asarray(u, dtype=float)
        a, m, b = self._a, self._m, self._b
        Fm = (m - a) / (b - a)

        x = np.empty_like(u, dtype=float)
        left = u <= Fm
        right = ~left

        x[left] = a + np.sqrt(u[left] * (b - a) * (m - a))
        x[right] = b - np.sqrt((1.0 - u[right]) * (b - a) * (b - m))
        return x


class Triangular(SciPyWrapper):
    """Convenience TrueMeasure wrapper around TriangularDistribution."""

    def __init__(self, sampler, c=0.5, loc=0.0, scale=1.0) -> None:
        super().__init__(
            sampler=sampler,
            scipy_distribs=TriangularDistribution(c=c, loc=loc, scale=scale),
        )
