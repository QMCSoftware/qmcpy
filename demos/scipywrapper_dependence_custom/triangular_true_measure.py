import numpy as np

from qmcpy import SciPyWrapper
from qmcpy.util import ParameterError


class TriangularUserDistribution:
    """
    Triangular distribution that mirrors `scipy.stats.triang`.

    Support: [loc, loc + scale]
    Mode: loc + c * scale, where 0 < c < 1.
    Implements ppf and pdf so SciPyWrapper can use it.
    """

    def __init__(self, c=0.5, loc=0.0, scale=1.0):
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

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, m, b = self._a, self._m, self._b
        out = np.zeros_like(x, dtype=float)

        left = (x >= a) & (x < m)
        right = (x >= m) & (x <= b)

        out[left] = 2.0 * (x[left] - a) / ((b - a) * (m - a))
        out[right] = 2.0 * (b - x[right]) / ((b - a) * (b - m))
        return out

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        a, m, b = self._a, self._m, self._b
        Fm = (m - a) / (b - a)

        x = np.empty_like(u, dtype=float)
        left = u <= Fm
        right = ~left

        x[left] = a + np.sqrt(u[left] * (b - a) * (m - a))
        x[right] = b - np.sqrt((1.0 - u[right]) * (b - a) * (b - m))
        return x


class BadTriangularDistribution:
    """
    Intentionally broken distribution to trigger SciPyWrapper warnings.
    Only for demo use. Do not export from qmcpy.true_measure.
    """

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        return np.sin(np.pi * u)  # non-monotone on purpose

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.ones_like(x)  # not normalized on purpose


class TriangularTrueMeasure(SciPyWrapper):
    """
    Convenience true measure for the custom triangular marginal example.

    Example:
    >>> tm = TriangularTrueMeasure(
    ...     sampler=DigitalNetB2(1, seed=7),
    ...     c=0.3,
    ...     loc=-1.0,
    ...     scale=2.0,
    ... )
    >>> tm(4).shape
    (4, 1)
    """

    def __init__(self, sampler, c=0.5, loc=0.0, scale=1.0):
        dist = TriangularUserDistribution(c=c, loc=loc, scale=scale)
        super().__init__(sampler=sampler, scipy_distribs=dist)
