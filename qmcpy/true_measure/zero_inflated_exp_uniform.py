import warnings

import numpy as np

from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper


class _ZeroInflatedExponential:
    """
    One-dimensional zero-inflated exponential distribution.

    This distribution has probability mass ``p_zero`` at zero and an
    exponential distribution with rate ``lam`` on positive values.

    It implements ``ppf`` so it can be passed to ``SciPyWrapper`` as a
    custom univariate marginal.
    """

    def __init__(self, p_zero=0.4, lam=1.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")

        self.p_zero = float(p_zero)
        self.lam = float(lam)

    def ppf(self, u):
        """
        Generalized inverse CDF of the zero-inflated exponential.

        SciPyWrapper supplies one coordinate at a time. For example:

            sampler output: (n, 1)
            ppf input:      (n,)
        """
        u = np.asarray(u, dtype=float)

        # Values up to p_zero map to the point mass at X = 0.
        x = np.zeros_like(u, dtype=float)
        mask_exp = u > self.p_zero

        # Rescale the remaining values to (0, 1), then use the
        # exponential inverse CDF.
        if np.any(mask_exp):
            u_rescaled = (u[mask_exp] - self.p_zero) / (
                1.0 - self.p_zero
            )
            u_rescaled = np.clip(
                u_rescaled,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
            x[mask_exp] = -np.log1p(-u_rescaled) / self.lam

        return x


class _DeprecatedZeroInflatedExpUniform2D:
    """
    Adapter for the deprecated two-dimensional ``y_split`` construction.
    """

    dim = 2

    def __init__(self, p_zero=0.4, lam=1.5, y_split=0.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")
        if not (0.0 < y_split < 1.0):
            raise ParameterError("y_split must be in (0,1).")

        self.p_zero = float(p_zero)
        self.lam = float(lam)
        self.y_split = float(y_split)

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        positive = u[..., 0] > self.p_zero

        t = np.empty_like(u, dtype=float)
        if np.any(positive):
            u_rescaled = (u[..., 0][positive] - self.p_zero) / (
                1.0 - self.p_zero
            )
            u_rescaled = np.clip(
                u_rescaled,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
            exp_values = -np.log1p(-u_rescaled) / self.lam
        else:
            exp_values = np.array([], dtype=float)

        t[..., 0] = 0.0
        t[..., 0][positive] = exp_values
        t[..., 1] = np.where(
            positive,
            self.y_split + (1.0 - self.y_split) * u[..., 1],
            self.y_split * u[..., 1],
        )
        return t

    def logpdf(self, x):
        return np.zeros(np.asarray(x).shape[:-1], dtype=float)


class ZeroInflatedExpUniform(SciPyWrapper):
    """
    One-dimensional zero-inflated exponential true measure.

    The ``y_split`` keyword is retained temporarily for backward
    compatibility with the deprecated two-dimensional construction.

    Examples
    --------
    Without replications:

    >>> from qmcpy import DigitalNetB2, ZeroInflatedExpUniform
    >>> tm = ZeroInflatedExpUniform(
    ...     DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
    ... )
    >>> x = tm(8)
    >>> x
    array([[0.        ],
           [0.76621559],
           [0.        ],
           [0.18405583],
           [0.08112272],
           [1.19997153],
           [0.        ],
           [0.33259467]])
    >>> x.shape
    (8, 1)
    >>> bool((x >= 0).all())
    True

    With independent replications:

    >>> tm = ZeroInflatedExpUniform(
    ...     DigitalNetB2(1, seed=7, replications=2),
    ...     p_zero=0.4,
    ...     lam=1.5,
    ... )
    >>> x = tm(8)
    >>> x
    array([[[0.51197024],
            [0.        ],
            [2.54258665],
            [0.03368876],
            [0.2192598 ],
            [0.        ],
            [0.85384192],
            [0.        ]],
    <BLANKLINE>
           [[1.3024994 ],
            [0.03378461],
            [0.20489897],
            [0.        ],
            [0.58638285],
            [0.        ],
            [0.35227285],
            [0.        ]]])
    >>> x.shape
    (2, 8, 1)
    >>> bool((x >= 0).all())
    True
    """

    def __init__(self, sampler, p_zero=0.4, lam=1.5, y_split=None):
        if y_split is not None:
            warnings.warn(
                "`y_split` is deprecated. The 2D zero-inflated "
                "exponential-uniform construction is retained only for "
                "backward compatibility. Prefer the 1D "
                "ZeroInflatedExpUniform interface.",
                DeprecationWarning,
                stacklevel=2,
            )

            if sampler.d == 2:
                scipy_distribs = _DeprecatedZeroInflatedExpUniform2D(
                    p_zero=p_zero,
                    lam=lam,
                    y_split=y_split,
                )
                self._deprecated_2d_y_split = True
            elif sampler.d == 1:
                scipy_distribs = _ZeroInflatedExponential(
                    p_zero=p_zero,
                    lam=lam,
                )
                self._deprecated_2d_y_split = False
            else:
                raise DimensionError(
                    "ZeroInflatedExpUniform with deprecated y_split requires "
                    "a one- or two-dimensional sampler."
                )
        else:
            if sampler.d != 1:
                raise DimensionError(
                    "ZeroInflatedExpUniform requires a one-dimensional sampler."
                )

            scipy_distribs = _ZeroInflatedExponential(
                p_zero=p_zero,
                lam=lam,
            )
            self._deprecated_2d_y_split = False

        super().__init__(
            sampler=sampler,
            scipy_distribs=scipy_distribs,
        )

        self.parameters = ["p_zero", "lam"]
        self.p_zero = float(p_zero)
        self.lam = float(lam)
        self.y_split = y_split
        if self._deprecated_2d_y_split:
            self.parameters.append("y_split")
            self.range = np.array([[0.0, np.inf], [0.0, 1.0]])

    def _spawn(self, sampler, dimension):
        if self._deprecated_2d_y_split:
            if dimension != 2:
                raise DimensionError(
                    "Deprecated y_split construction requires dimension 2."
                )
            return ZeroInflatedExpUniform(
                sampler=sampler,
                p_zero=self.p_zero,
                lam=self.lam,
                y_split=self.y_split,
            )

        return super()._spawn(sampler, dimension)
