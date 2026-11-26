from typing import Union
import numpy as np

from ..util import DimensionError
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..true_measure.abstract_true_measure import AbstractTrueMeasure
from .scipy_wrapper import SciPyWrapper
from ..discrete_distribution import DigitalNetB2


class _UniformTriangleAdapter:
    r"""Uniform on triangle $T = \{(x, y): 0 \le y \le x \le 1\}$

    Exact transform:
      $$u_1, u_2 \sim U(0, 1)$$
      $$x = \sqrt{u_1}$$
      $$y = u_2 x$$
    """

    def __init__(self):
        self.dim = 2
        self._log_density = float(np.log(2.0))  # area(T) = 1/2, so density = 2

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != 2:
            raise DimensionError(f"Expected last axis 2, got {u.shape[-1]}")

        u1 = u[..., 0]
        u2 = u[..., 1]

        x = np.sqrt(u1)
        y = u2 * x

        out = np.empty(u.shape, dtype=float)
        out[..., 0] = x
        out[..., 1] = y
        return out

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != 2:
            raise DimensionError(f"Expected last axis 2, got {x.shape[-1]}")

        xx = x[..., 0]
        yy = x[..., 1]

        inside = (xx >= 0.0) & (xx <= 1.0) & (yy >= 0.0) & (yy <= xx)
        out = np.full(x.shape[:-1], -np.inf, dtype=float)
        out[inside] = self._log_density
        return out


class UniformTriangle(SciPyWrapper):
    r"""Uniform distribution on the triangle $\{(x, y): 0 \le y \le x \le 1\}$.

    Examples:
        >>> tm = UniformTriangle(sampler=DigitalNetB2(2, seed=7))
        >>> x = tm(4)
        >>> x
        array([[0.84948429, 0.7772399 ],
               [0.40429635, 0.17370534],
               [0.99335923, 0.03413563],
               [0.65541327, 0.36622096]])
        >>> x.shape
        (4, 2)
        >>> bool(np.all(x[:, 1] <= x[:, 0]))
        True

        With independent replications:

        >>> tm_rep = UniformTriangle(DigitalNetB2(2, seed=7, replications=2))
        >>> tm_rep(4)
        array([[[0.78838045, 0.54833346],
                [0.23777138, 0.04996095],
                [0.92542139, 0.29275141],
                [0.53891021, 0.45310118]],
        <BLANKLINE>
               [[0.28115996, 0.07354063],
                [0.88085513, 0.63776346],
                [0.6527037 , 0.61405078],
                [0.86506152, 0.03878966]]])
    """

    def __init__(self, sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure]) -> None:
        """Initialize a UniformTriangle true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): A
                2-dimensional sampler generating unit-cube samples to be
                transformed to the triangle.
        """
        super().__init__(sampler=sampler, scipy_distribs=_UniformTriangleAdapter())
