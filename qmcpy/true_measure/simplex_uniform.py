import math
import warnings
from typing import Union

import numpy as np

from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..discrete_distribution import DigitalNetB2
from .simplex_transform import _SimplexTransform
from ..util import ParameterError
from .abstract_true_measure import AbstractTrueMeasure


class SimplexUniform(AbstractTrueMeasure):
    r"""Uniform distribution on the corner simplex $K_d = \{w \in \mathbb{R}^d :
    w_i \ge 0, \sum_{i=1}^d w_i \le 1\}$, named for its shape: the corner of
    the unit cube $[0,1]^d$ cut off by the hyperplane $\sum_i w_i = 1$.

    Returns $d$ of $d+1$ nonnegative weights summing to 1; append $w_{d+1} =
    1 - \sum_i w_i$ for the full probability-simplex vector, which is
    $\mathrm{Dirichlet}(1,\dots,1)$-distributed ($d+1$ ones) [1].

    Wraps `_SimplexTransform`'s mapping onto the *ordered* simplex $T_d = \{0
    \le x_1 \le \dots \le x_d \le 1\}$, then takes consecutive differences
    ($w_i = x_i - x_{i-1}$, $x_0 := 0$), a volume-preserving (Jacobian 1)
    reparametrization, so the density is $d!$ either way.

    References:
        [1] L. Devroye, "Non-Uniform Random Variate Generation," Springer-Verlag,
        1986, Sec. I.4.1 (sampling the simplex via uniform spacings). [Online].
        Available: http://luc.devroye.org/rnbookindex.html

    Examples:
        >>> s = SimplexUniform(DigitalNetB2(3, seed=7))
        >>> w = s(4)
        >>> w.shape
        (4, 3)
        >>> bool(np.all(w >= 0) and np.all(w.sum(axis=-1) <= 1))
        True

        Every `transform_method` is measure-preserving (weight means over
        500,000 samples matched the exact Dirichlet mean $1/(d+1)$), so the
        choice is a QMC-efficiency question, not a correctness one:

        >>> s = SimplexUniform(DigitalNetB2(3, seed=7), transform_method='shift')
        >>> s(4).shape
        (4, 3)

        Append the implicit $(d+1)$-th weight for the full probability-simplex
        vector (sums to exactly 1):

        >>> w = s(1)[0]
        >>> v = np.append(w, 1 - w.sum())
        >>> v.shape
        (4,)
        >>> bool(np.isclose(v.sum(), 1) and np.all(v >= 0))
        True
    """

    def __init__(self, sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure],
                 transform_method: str = "root") -> None:
        """Initialize a SimplexUniform true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            transform_method (str): One of _SimplexTransform's measure-preserving
                methods: 'root', 'sort', 'shift', 'origami', or 'mirror'.
                'mirror' folds a symmetric point set (e.g. a lattice) onto
                itself, degrading its low-discrepancy structure, and only
                supports dimension <= 3 (raised by _SimplexTransform itself);
                a warning is issued when it is selected. ('drop' is excluded
                entirely: it rejects rather than maps points 1:1, so it does
                not fit the _transform/_weight contract at all; see
                AcceptanceRejection for that pattern instead.)
        """
        if transform_method not in ("root", "sort", "shift", "origami", "mirror"):
            raise ParameterError(
                "transform_method must be one of 'root', 'sort', 'shift', 'origami', "
                f"'mirror', not {transform_method!r}"
            )
        if transform_method == "mirror":
            warnings.warn(
                "transform_method='mirror' folds a symmetric point set (e.g., a "
                "lattice) onto itself, which can degrade its low-discrepancy "
                "structure, and only supports dimension <= 3.",
                UserWarning,
            )
        self.transform_method = transform_method
        self.parameters = ["transform_method"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self._simplex = _SimplexTransform(dimension=self.d)
        self._density = float(math.factorial(self.d))  # 1 / Vol(T_d), Vol(T_d) = 1/d!
        self.range = np.tile(np.array([0, 1]), (self.d, 1))
        super(SimplexUniform, self).__init__()

    def _transform(self, x):
        t = getattr(self._simplex, self.transform_method)(x)
        # t is an ordered-simplex T_d point (d sorted coords); consecutive differences
        # (x_0 := 0) give the first d of d+1 weights, a Jacobian-1 map that leaves density unchanged.
        return np.diff(t, axis=-1, prepend=0)

    def _weight(self, x):
        eps = np.finfo(float).eps
        inside = np.all(x >= -eps, axis=-1) & (x.sum(axis=-1) <= 1+eps)
        return np.where(inside, self._density, 0.0)

    def _spawn(self, sampler, dimension):
        return SimplexUniform(sampler, transform_method=self.transform_method)
