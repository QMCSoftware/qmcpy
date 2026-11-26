from typing import Union

import numpy as np

from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import ParameterError
from .abstract_true_measure import AbstractTrueMeasure
from .scipy_wrapper import SciPyWrapper


class TriangularDistribution:
    """Triangular distribution matching scipy.stats.triang behavior.

    Support: [loc, loc + scale]
    Mode: loc + c*scale, with 0 < c < 1
    Provides ppf and pdf for SciPyWrapper custom-marginal usage.
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
    r"""Convenience TrueMeasure wrapper around TriangularDistribution.

    Examples:
        >>> import numpy as np
        >>> from qmcpy import DigitalNetB2
        >>> tm = Triangular(DigitalNetB2(2, seed=7), c=0.25, loc=1.5, scale=3)
        >>> samples = tm(4)
        >>> np.round(samples, 7)
        array([[3.1292189, 3.7423366],
               [2.1064445, 2.5378909],
               [4.2010807, 1.7780623],
               [2.5377498, 2.7742112]])

        With independent replications:

        >>> tm_rep = Triangular(
        ...     DigitalNetB2(2, seed=7, replications=2),
        ...     c=0.25, loc=1.5, scale=3,
        ... )
        >>> samples = tm_rep(4)
        >>> np.round(samples, 7)
        array([[[2.9016947, 3.0663865],
                [1.8566571, 2.1875857],
                [3.5154859, 2.3518197],
                [2.3114762, 3.4632831]],
        <BLANKLINE>
               [[1.9217399, 2.2674096],
                [3.2701073, 3.1351508],
                [2.531663 , 3.8677555],
                [3.196634 , 1.8176331]]])
    """

    def __init__(
        self,
        sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure],
        c: float = 0.5,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        """Initialize a triangular true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                A discrete distribution or transform whose range is the unit cube.
            c (float): Relative mode position, strictly between zero and one.
            loc (float): Lower endpoint of the support.
            scale (float): Positive support width; the upper endpoint is loc + scale.
        """
        super().__init__(
            sampler=sampler,
            scipy_distribs=TriangularDistribution(c=c, loc=loc, scale=scale),
        )
