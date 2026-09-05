import numpy as np

from ..util import DimensionError
from .scipy_wrapper import SciPyWrapper
from ..discrete_distribution import DigitalNetB2


class _UniformTriangleAdapter:
    """Uniform on triangle T = {(x, y): 0 <= y <= x <= 1}

    Exact transform:
      u1, u2 ~ U(0, 1) x = sqrt(u1) y = u2 * x
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
    """Uniform distribution on the triangle {(x, y): 0 <= y <= x <= 1}.

    Examples:
    >>> tm = UniformTriangle(sampler=DigitalNetB2(2, seed=7))
    >>> x = tm(4)
    >>> x.shape
    (4, 2)
    >>> bool(np.all(x[:, 1] <= x[:, 0]))
    True
    """

    def __init__(self, sampler):
        super().__init__(sampler=sampler, scipy_distribs=_UniformTriangleAdapter())
