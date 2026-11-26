import math
import warnings
from typing import Union

import numpy as np

from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
from .abstract_true_measure import AbstractTrueMeasure


class SimplexUniform(AbstractTrueMeasure):
    r"""Uniform distribution on a corner or ordered simplex.

    The default corner simplex is $K_d = \{w \in \mathbb{R}^d : w_i \ge 0,
    \sum_{i=1}^d w_i \le 1\}$, named for its shape: the corner of the unit cube
    $[0,1]^d$ cut off by the hyperplane $\sum_i w_i = 1$. Its $d$ nonnegative
    coordinates sum to at most 1. Append $w_{d+1} = 1 - \sum_i w_i$ for the
    full probability-simplex vector, which is
    $\mathrm{Dirichlet}(1,\dots,1)$-distributed ($d+1$ ones) [1]. The ordered
    simplex is $T_d = \{0 \le x_1 \le \dots \le x_d \le 1\}$.

    The static methods `drop`, `root`, `sort`, `shift`, `origami`, and `mirror`
    map supplied cube points onto $T_d$ and may be called without constructing
    a measure, for example `SimplexUniform.sort(points)`. The direct methods
    always return ordered coordinates. With `simplex_type='ordered'`, a
    constructed measure returns those coordinates directly. With
    `simplex_type='corner'`, consecutive differences are returned
    ($w_i = x_i - x_{i-1}$, $x_0 := 0$). This reparametrization has Jacobian 1,
    so the uniform density is $d!$ for either simplex.

    Here `transform_method='shift'` means the arbitrary-dimensional
    $I^d \to K_d \to T_d$ construction in [3], including when $d=2$. It is
    distinct pointwise from the original two-dimensional Shift in [2], whose
    authors reported no higher-dimensional generalization at that time.

    References:
        [1] L. Devroye, "Non-Uniform Random Variate Generation," Springer-Verlag,
        1986, Sec. V.2.1 (uniform spacings). [Online].
        Available: http://luc.devroye.org/rnbookindex.html

        [2] T. Pillards and R. Cools, "Transforming low-discrepancy sequences from
        a cube to a simplex," *Journal of Computational and Applied Mathematics*,
        vol. 174, no. 1, pp. 29-42, 2005.

        [3] T. Pillards, "Quasi-Monte Carlo integration over a simplex and the
        entire space," Ph.D. thesis, KU Leuven, 2006. [Online]. Available:
        https://www.cs.kuleuven.be/publicaties/doctoraten/tw/TW2006_05.pdf

    Examples:
        >>> s = SimplexUniform(DigitalNetB2(3, seed=7))
        >>> w = s(4)
        >>> w
        array([[0.49677051, 0.28841988, 0.09289854],
               [0.08172615, 0.3489796 , 0.36260659],
               [0.11491863, 0.01307957, 0.17307836],
               [0.34855887, 0.4160591 , 0.17501823]])
        >>> bool(np.all(w >= 0) and np.all(w.sum(axis=-1) <= 1))
        True

        `replications=None` omits a replication axis, while explicit values
        retain it, including `replications=1`:

        >>> for replications in (None, 1, 2):
        ...     measure = SimplexUniform(
        ...         DigitalNetB2(2, seed=7, replications=replications)
        ...     )
        ...     print(f"{replications}: {np.round(measure(1), 3).tolist()}")
        None: [[0.69, 0.266]]
        1: [[[0.69, 0.266]]]
        2: [[[0.518, 0.316]], [[0.04, 0.471]]]

        Each coordinate's moments follow the symmetric Dirichlet$(1,\dots,1)$
        marginal/pairwise formulas $E[W_i] = \frac{1}{d+1}$,
        $\mathrm{Var}(W_i) = \frac{d}{(d+1)^2(d+2)}$,
        $\mathrm{Cov}(W_i,W_j) = -\frac{1}{(d+1)^2(d+2)}$ ($i \ne j$). Unlike
        `Kumaraswamy`'s independent coordinates, this covariance is dense, not
        diagonal: the $d+1$ weights share a fixed budget summing to 1.

        >>> s2 = SimplexUniform(DigitalNetB2(2, seed=7))
        >>> s2  # doctest: +NORMALIZE_WHITESPACE
        SimplexUniform (AbstractTrueMeasure)
            transform_method root
            simplex_type    corner
            mean            [0.333 0.333]
            variance        [0.056 0.056]
            standard_deviation [0.236 0.236]
            covariance      [[ 0.056 -0.028]
                             [-0.028  0.056]]

        For the corner representation, every `transform_method` is measure-
        preserving (weight means over 500,000 samples matched the exact
        Dirichlet mean $1/(d+1)$), so the choice is a QMC-efficiency question,
        not a correctness one:

        >>> s = SimplexUniform(DigitalNetB2(3, seed=7), transform_method='shift')
        >>> s(4)
        array([[0.21089174, 0.3556335 , 0.23307505],
               [0.06324979, 0.11575659, 0.32026034],
               [0.80289593, 0.08582128, 0.00909724],
               [0.15195339, 0.25510833, 0.4225583 ]])

        Append the implicit $(d+1)$-th weight for the full probability-simplex
        vector (sums to exactly 1):

        >>> w = s(1)[0]
        >>> v = np.append(w, 1 - w.sum())
        >>> v
        array([0.21089174, 0.3556335 , 0.23307505, 0.2003997 ])
        >>> bool(np.isclose(v.sum(), 1) and np.all(v >= 0))
        True

        Request the ordered-simplex coordinates directly:

        >>> ordered = SimplexUniform(
        ...     DigitalNetB2(3, seed=7), simplex_type='ordered'
        ... )(2)
        >>> ordered
        array([[0.49677051, 0.78519039, 0.87808893],
               [0.08172615, 0.43070574, 0.79331233]])
        >>> bool(np.all(np.diff(ordered, axis=-1) >= 0))
        True
    """

    _TRANSFORM_METHODS = ("root", "sort", "shift", "origami", "mirror")
    _SIMPLEX_TYPES = ("corner", "ordered")

    @staticmethod
    def _validate_options(transform_method: str, simplex_type: str) -> None:
        """Validate transform options and warn about Mirror's limitations."""
        if transform_method not in SimplexUniform._TRANSFORM_METHODS:
            raise ParameterError(
                "transform_method must be one of 'root', 'sort', 'shift', 'origami', "
                f"'mirror', not {transform_method!r}"
            )
        if simplex_type not in SimplexUniform._SIMPLEX_TYPES:
            raise ParameterError(
                "simplex_type must be either 'corner' or 'ordered', "
                f"not {simplex_type!r}"
            )
        if transform_method == "mirror":
            warnings.warn(
                "transform_method='mirror' folds a symmetric point set (e.g., a "
                "lattice) onto itself, which can degrade its low-discrepancy "
                "structure, and only supports dimension <= 3.",
                UserWarning,
            )

    @staticmethod
    def _validate_points(points: np.ndarray) -> np.ndarray:
        """Return finite cube points with a nonempty final coordinate axis."""
        points = np.asarray(points)
        if (
            points.ndim == 0
            or not np.issubdtype(points.dtype, np.number)
            or np.iscomplexobj(points)
        ):
            raise ParameterError(
                "points must be a real numeric array with shape (..., d)"
            )
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[-1] < 1:
            raise ParameterError("points must have at least one coordinate")
        points = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(points)) or np.any((points < 0) | (points > 1)):
            raise ParameterError("points must contain finite values in [0, 1]")
        return points

    @staticmethod
    def drop(points: np.ndarray) -> np.ndarray:
        r"""Keep only cube points inside the ordered simplex $T_d$.

        Drop is simple but retains only about $1/d!$ of the input points.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Points in the ordered simplex $T_d$, shape (M, d).
                Leading batch axes are flattened because each batch may retain a
                different number of points.

        Examples:
            >>> import numpy as np
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> result = SimplexUniform.drop(points)
            >>> result
            array([[0.3, 0.7]])
        """
        # Check if points satisfy x1 <= x2 <= ... <= xd
        points = SimplexUniform._validate_points(points)
        mask = np.all(points[..., :-1] <= points[..., 1:], axis=-1)
        return points.reshape(-1, points.shape[-1])[mask.reshape(-1)]

    @staticmethod
    def sort(points: np.ndarray) -> np.ndarray:
        r"""Sort each point's coordinates into the ordered simplex $T_d$.

        Sort is fast, continuous, and retains every point, but is not injective:
        permutations have the same image. Following [2], the output is a
        multiset; each occurrence keeps its original QMC weight.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> result = SimplexUniform.sort(points)
            >>> result
            array([[0.3, 0.7],
                   [0.4, 0.8]])
            >>> SimplexUniform.sort(np.array([[0.3, 0.7], [0.7, 0.3]]))
            array([[0.3, 0.7],
                   [0.3, 0.7]])
        """
        points = SimplexUniform._validate_points(points)
        return np.sort(points, axis=-1)

    @staticmethod
    def root(points: np.ndarray) -> np.ndarray:
        r"""Map cube points into $T_d$ with the inverse-CDF recurrence.

        Root is continuous, parameter-free, and bijective on the cube interior,
        though boundary images may coincide; see [2], Sec. 2.5, and [3], Sec. 4.3.5.
        For input $(x_1,\dots,x_d)$, the output $(y_1,\dots,y_d)$ satisfies

        $$y_d := x_d^{1/d}, \qquad y_i := y_{i+1} \cdot x_i^{1/i} \quad \text{for } i = d-1, \dots, 1.$$

        Root has highly nonuniform displacement: points near $x_d=0$ move far
        more than points near $x_d=1$.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> SimplexUniform.root(np.array([0.5, 0.01]))
            array([[0.05, 0.1 ]])
            >>> np.round(SimplexUniform.root(np.array([0.5, 0.99])), 3)
            array([[0.497, 0.995]])

            The direct transform methods have no `replications` parameter; leading batch
            axes (the "..." in ``shape (..., d)``) pass through unchanged. For
            example, this input contains two replications with two points each:

            >>> batch = np.array([[[0.5, 0.01], [0.5, 0.99]], [[0.3, 0.7], [0.8, 0.4]]])
            >>> batch.shape
            (2, 2, 2)
            >>> np.round(SimplexUniform.root(batch), 3)
            array([[[0.05 , 0.1  ],
                    [0.497, 0.995]],
            <BLANKLINE>
                   [[0.251, 0.837],
                    [0.506, 0.632]]])
        """
        points = SimplexUniform._validate_points(points)
        d = points.shape[-1]
        y = np.empty_like(points)
        y[..., -1] = points[..., -1] ** (1.0 / d)
        for i in range(d - 2, -1, -1):
            y[..., i] = y[..., i + 1] * points[..., i] ** (1.0 / (i + 1))
        return y

    @staticmethod
    def mirror(points: np.ndarray) -> np.ndarray:
        r"""Keep points in $T_d$ fixed and reflect every other point into it.

        Based on [2], Sec. 2.3, and [3], Sec. 4.3.3. This implementation covers
        the references' explicit formulas for $d \in \{2, 3\}$ and uses the
        identity for $d=1$. Mirror is fast but discontinuous and folds half of
        a centrally symmetric point set, such as a lattice, onto the other half.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d),
                $d \in \{1, 2, 3\}$

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Raises:
            NotImplementedError: if the points have dimension greater than 3

        Examples:
            >>> import numpy as np
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> SimplexUniform.mirror(points)
            array([[0.3, 0.7],
                   [0.2, 0.6]])
        """
        y = SimplexUniform._validate_points(points).copy()
        d = y.shape[-1]
        flat = y.reshape(-1, d)
        if d == 1:
            return y
        if d == 2:
            swap = flat[:, 0] > flat[:, 1]
            flat[swap] = 1.0 - flat[swap]
            return y
        if d == 3:
            x1, x2, x3 = flat[:, 0].copy(), flat[:, 1].copy(), flat[:, 2].copy()
            m = x3 <= x1
            x1[m], x2[m], x3[m] = 1.0 - x1[m], 1.0 - x2[m], 1.0 - x3[m]
            m = x3 <= x2
            x2[m], x3[m] = 1.0 - x2[m] + x1[m], 1.0 - x3[m] + x1[m]
            m = x2 <= x1
            x1_new, x2_new = x3[m] - x1[m], x3[m] - x2[m]
            x1[m], x2[m] = x1_new, x2_new
            flat[...] = np.stack([x1, x2, x3], axis=-1)
            return y
        raise NotImplementedError(
            "Transformation Mirror is implemented only for dimensions 1-3; "
            "see [2], Sec. 2.3, and [3], Sec. 4.3.3."
        )

    @staticmethod
    def origami(points: np.ndarray, base: int = 2, depth: int = 1) -> np.ndarray:
        r"""Apply Sort recursively from a fine grid to the whole cube.

        Based on [2], Sec. 2.4, and [3], Sec. 4.3.4. Choosing a base $b$ and
        depth $m$, the finest grid has $M=b^m$ cells per axis. Origami applies
        Sort within its $M^d$ cells, then at resolutions $M/b,\dots,b,1$; the
        final pass is global Sort. It is discontinuous but preserves the sizes
        of elementary intervals [3], Sec. 2.3.5. QMCPy defines `depth=0` as
        Sort and assigns coordinates equal to 1 to the final grid cell.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)
            base (int): grid subdivisions per level, $b \ge 2$
            depth (int): number of levels above the base grid, $m \ge 0$

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> SimplexUniform.origami(np.array([0.9, 0.3]), base=2, depth=1)
            array([[0.4, 0.8]])
        """
        x = SimplexUniform._validate_points(points)
        if (
            isinstance(base, bool)
            or not isinstance(base, (int, np.integer))
            or base < 2
        ):
            raise ParameterError("base must be an integer greater than or equal to 2")
        if (
            isinstance(depth, bool)
            or not isinstance(depth, (int, np.integer))
            or depth < 0
        ):
            raise ParameterError("depth must be a nonnegative integer")
        b = int(base)
        y = x.copy()
        for n in (b ** k for k in range(depth, -1, -1)):
            # Put y=1 in the last cell; an out-of-range index can leave the
            # point unsorted and produce a negative corner-simplex gap.
            cell = np.minimum(np.floor(n * y), n - 1)
            frac = n * y - cell
            frac.sort(axis=-1)
            y = (cell + frac) / n
        return y

    @staticmethod
    def shift(points: np.ndarray) -> np.ndarray:
        r"""Map $I^d$ through $K_d$ to $T_d$ with the Shift in [3].

        The original Shift is a two-dimensional piecewise-linear map; [2],
        Sec. 2.6, reports no higher-dimensional extension. This method uses
        the thesis's arbitrary-dimensional $I^d\to K_d\to T_d$
        construction [3], Sec. 4.3.6, including for $d=2$. Both preserve the
        uniform measure, but the original maps $(0.8,0.4)$ to $(0.5,0.7)$ and
        this map gives $(0.6,0.8)$. 

        For the construction in [3], sorting the input ascending, then for
        $k = 1, \dots, d-1$ subtracting $(d-k)/(d-k+1)$ times the gap between
        sorted coordinates $k-1$ and $k$ from every sorted coordinate at or
        after position $k$ maps the cube onto $K_d$; undoing the sort and
        taking cumulative sums maps $K_d$ onto $T_d$. Shift is continuous and
        parameter-free. Its 2D intervals are more compact than Root's in the
        illustration in [3], Sec. 4.4.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> SimplexUniform.shift(points)
            array([[0.15, 0.7 ],
                   [0.6 , 0.8 ]])
        """
        points = SimplexUniform._validate_points(points)
        d = points.shape[-1]
        if d == 1:
            return points.copy()
        order = np.argsort(points, axis=-1)
        x = np.take_along_axis(points, order, axis=-1)
        for j in range(d - 1):
            prev = x[..., j - 1] if j else 0.0
            gap = x[..., j] - prev
            x[..., j:] -= (((d - j - 1) / (d - j)) * gap)[..., None]
        a = np.take_along_axis(x, np.argsort(order, axis=-1), axis=-1)
        return np.cumsum(a, axis=-1)

    def __init__(
        self,
        sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure],
        transform_method: str = "root",
        simplex_type: str = "corner",
    ) -> None:
        """Initialize a SimplexUniform true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.

                Its dimension must be <= 170: the density d! exceeds float64's
                range above that, raising ParameterError at construction.
            transform_method (str): One of `SimplexUniform`'s measure-preserving
                methods [2, 3]:

                - 'root' (default)
                - 'sort', which maps coordinate permutations to the same point
                - 'shift', the arbitrary-dimensional construction through
                  $K_d$ in [3], not the original two-dimensional formula from
                  [2]
                - 'origami', which is not injective and uses the defaults
                  `base=2` and `depth=1`
                - 'mirror', which is not injective, folds a symmetric point set
                  (e.g. a lattice) onto itself, and only supports dimension <= 3;
                  selecting it issues a warning

                For the non-injective methods, all occurrences of a repeated
                image are retained as a multiset so that each keeps its original
                QMC weight. Removing repeated images would change the quadrature
                rule.

                - 'drop' is excluded because it rejects points and changes the
                  sample count, so it does not fit the `_transform`/`_weight`
                  contract; see `AcceptanceRejection` for that pattern instead.
            simplex_type (str): Simplex representation to return:

                - 'corner' (default) returns nonnegative coordinates whose sum
                  is at most 1
                - 'ordered' returns nondecreasing coordinates in $[0,1]$
        """
        self._validate_options(transform_method, simplex_type)
        self.transform_method = transform_method
        self.simplex_type = simplex_type
        self.parameters = [
            "transform_method",
            "simplex_type",
            "mean",
            "variance",
            "standard_deviation",
            "covariance",
        ]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        try:
            self._density = float(math.factorial(self.d))  # 1 / Vol(T_d), Vol(T_d) = 1/d!
        except OverflowError:
            raise ParameterError(
                f"SimplexUniform's density d! is not representable as a float64 "
                f"for dimension {self.d} (float64 overflows above d=170)."
            )
        self.range = np.tile(np.array([0, 1]), (self.d, 1))
        d = self.d
        denominator = (d + 1) ** 2 * (d + 2)
        if self.simplex_type == "corner":
            # These are the first d coordinates of a symmetric
            # Dirichlet(1,...,1) vector with d+1 parts.
            mean = np.full(d, 1.0 / (d + 1))
            variance = np.full(d, d / denominator)
            covariance = np.full((d, d), -1.0 / denominator)
            np.fill_diagonal(covariance, variance)
        else:
            # Ordered-simplex coordinates have the distribution of the order
            # statistics of d independent standard uniform random variables.
            indices = np.arange(1, d + 1, dtype=float)
            mean = indices / (d + 1)
            variance = indices * (d - indices + 1) / denominator
            covariance = (
                np.minimum.outer(indices, indices)
                * (d - np.maximum.outer(indices, indices) + 1)
                / denominator
            )
        self._set_moments(
            mean=mean,
            variance=variance,
            standard_deviation=np.sqrt(variance),
            covariance=covariance,
        )
        super(SimplexUniform, self).__init__()

    def _transform(self, x):
        t = getattr(type(self), self.transform_method)(x)
        if self.simplex_type == "ordered":
            return t
        # Consecutive differences give the first d of d+1 weights. The map has
        # Jacobian 1, so it leaves the density unchanged.
        return np.diff(t, axis=-1, prepend=0)

    def _map_effective_range(self, input_range):
        bounds = self._broadcast_box(input_range, self.d)
        full_cube = self._broadcast_box(self.domain, self.d)
        if bounds is None or not np.array_equal(bounds, full_cube):
            return None
        return self.range

    @staticmethod
    def transform_points(
        x: np.ndarray,
        transform_method: str = "root",
        simplex_type: str = "corner",
    ) -> np.ndarray:
        r"""Map pre-generated cube points without constructing a measure.

        This method has no dimension limit because it does not compute the
        density $d!$. To compare transforms on identical draws, generate one
        point set and pass the same points or slices here.

        Args:
            x (np.ndarray): Points with shape `(..., d)`, each row in
                $[0,1]^d$. Not validated against any sampler; the caller is
                responsible for `x` being an appropriate input (e.g. a
                uniform point set) for `transform_method`.
            transform_method (str): One of `SimplexUniform`'s measure-
                preserving methods [2, 3]: 'root', 'sort', 'shift', 'origami', or
                'mirror' (same constraints and multiset semantics as
                `__init__`'s own Args).
            simplex_type (str): 'corner' (default) returns consecutive
                differences of the transformed ordered coordinates; 'ordered'
                returns those coordinates directly.

        Returns:
            np.ndarray: Points on the requested simplex, shape `(..., d)`.
                For `simplex_type='corner'`, append the implicit `(d+1)`-th
                weight for the full probability vector.

            >>> x = DigitalNetB2(3, seed=7).gen_samples(4)
            >>> w = SimplexUniform.transform_points(x)
            >>> w.shape
            (4, 3)
            >>> full = np.concatenate([w, 1 - w.sum(axis=-1, keepdims=True)], axis=-1)
            >>> bool(np.allclose(full.sum(axis=-1), 1) and np.all(full >= 0))
            True
            >>> ordered = SimplexUniform.transform_points(x, simplex_type='ordered')
            >>> bool(np.all(np.diff(ordered, axis=-1) >= 0))
            True
        """
        SimplexUniform._validate_options(transform_method, simplex_type)
        t = getattr(SimplexUniform, transform_method)(x)
        if simplex_type == "ordered":
            return t
        return np.diff(t, axis=-1, prepend=0)

    def _weight(self, x):
        eps = np.finfo(float).eps
        if self.simplex_type == "ordered":
            inside = (
                np.all((x >= -eps) & (x <= 1 + eps), axis=-1)
                & np.all(x[..., :-1] <= x[..., 1:] + eps, axis=-1)
            )
        else:
            inside = np.all(x >= -eps, axis=-1) & (x.sum(axis=-1) <= 1 + eps)
        return np.where(inside, self._density, 0.0)

    def _spawn(self, sampler, dimension):
        return SimplexUniform(
            sampler,
            transform_method=self.transform_method,
            simplex_type=self.simplex_type,
        )
