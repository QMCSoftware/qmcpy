r"""
Transformations of points from a unit hypercube onto the ordered simplex
$T_d = \{0 \le x_1 \le x_2 \le \dots \le x_d \le 1\}$.

Implements Drop, Sort, Mirror, Origami, Root, and Shift as described in [1],
with fuller derivations in Chapter 4 ("Transformations for a Simplex") of [2].

Authors: Larysa Matiukha and Sou-Cheng T. Choi
Date: February 6, 2026

Unit tests:
    python -W ignore -m unittest test.test_tm_true_measures.TestSimplexTransform -v

References:
    [1] T. Pillards and R. Cools, "Transforming low-discrepancy sequences from a cube
    to a simplex," *Journal of Computational and Applied Mathematics*, vol. 174, no. 1,
    pp. 29-42, 2005.

    [2] T. Pillards, "Quasi-Monte Carlo integration over a simplex and the entire
    space," Ph.D. thesis, KU Leuven, 2006. [Online]. Available:
    https://www.cs.kuleuven.be/publicaties/doctoraten/tw/TW2006_05.pdf
"""

import numpy as np

from ..util import ParameterError


class _SimplexTransform:
    r"""
    A class implementing various transformations from the unit cube to the ordered simplex $T_d$.

    This stateless helper transforms supplied points; it does not generate points.
    The simplex is $T_d = \{(x_1,\dots,x_d) \in \mathbb{R}^d : 0 \le x_1 \le \dots \le x_d \le 1\}$.

    Attributes:
        dimension (int): The dimension of the space
    """

    def __init__(self, dimension: int = 2):
        """
        Initialize the _SimplexTransform class.

        Args:
            dimension (int): The dimension of the space (default: 2)

        Examples:
            >>> _SimplexTransform(dimension=3).dimension
            3
        """
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
            or dimension < 1
        ):
            raise ParameterError("dimension must be a positive integer")
        self.dimension = int(dimension)

    def _validate_points(self, points: np.ndarray) -> np.ndarray:
        """Return finite cube points with a checked final coordinate axis."""
        points = np.asarray(points)
        if points.ndim == 0 or not np.issubdtype(points.dtype, np.number):
            raise ParameterError(
                "points must be a numeric array with shape (..., dimension)"
            )
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[-1] != self.dimension:
            raise ParameterError(
                f"points must have final dimension {self.dimension}, got {points.shape[-1]}"
            )
        points = np.asarray(points, dtype=float)
        if not np.all(np.isfinite(points)) or np.any((points < 0) | (points > 1)):
            raise ParameterError("points must contain finite values in [0, 1]")
        return points

    def drop(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Drop: Keep only points that fall inside the ordered simplex $T_d$.

        This is a straightforward but inefficient transformation. Only $1$ out
        of $d!$ points is kept in higher dimensions.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Points in the ordered simplex $T_d$, shape (M, d).
                Leading batch axes are flattened because each batch may retain a
                different number of points.

        Examples:
            >>> import numpy as np
            >>> transformer = _SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> result = transformer.drop(points)
            >>> result
            array([[0.3, 0.7]])
        """
        # Check if points satisfy x1 <= x2 <= ... <= xd
        points = self._validate_points(points)
        mask = np.all(points[..., :-1] <= points[..., 1:], axis=-1)
        return points.reshape(-1, self.dimension)[mask.reshape(-1)]

    def sort(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Sort: Sort the coordinates of each point.

        This is a fast, continuous transformation that recovers points lost by Drop.
        When we sort the coordinates of a point in $I^d$ (such that $x_i \le x_{i+1}$),
        we obtain a point in the simplex $T_d$.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> transformer = _SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> result = transformer.sort(points)
            >>> result
            array([[0.3, 0.7],
                   [0.4, 0.8]])
        """
        points = self._validate_points(points)
        return np.sort(points, axis=-1)

    def root(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Root: map points via the cumulative distribution function.

        Based on [1], Sec. 2.5, and [2], Sec. 4.3.5: a bijective, continuous
        transformation for any dimension $d$, with no free parameters.
        Writing the input as $(x_1,\dots,x_d)$, the output $(y_1,\dots,y_d)$ is

        $$y_d := x_d^{1/d}, \qquad y_i := y_{i+1} \cdot x_i^{1/i} \quad \text{for } i = d-1, \dots, 1.$$

        Root has highly nonuniform displacement: points near $x_d=0$ move far
        more than points near $x_d=1$.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> transformer = _SimplexTransform(dimension=2)
            >>> transformer.root(np.array([0.5, 0.01]))
            array([[0.05, 0.1 ]])
            >>> np.round(transformer.root(np.array([0.5, 0.99])), 3)
            array([[0.497, 0.995]])

            Leading batch axes (the "..." in ``shape (..., d)``) pass through
            unchanged, e.g. a (replications, portfolios, dimension) array as
            used elsewhere in this package:

            >>> batch = np.array([[[0.5, 0.01], [0.5, 0.99]], [[0.3, 0.7], [0.8, 0.4]]])
            >>> transformer.root(batch).shape
            (2, 2, 2)
        """
        points = self._validate_points(points)
        d = points.shape[-1]

        def _root(flat):
            y = np.empty_like(flat)
            y[..., d - 1] = flat[..., d - 1] ** (1.0 / d)
            for i in range(d - 2, -1, -1):
                y[..., i] = y[..., i + 1] * flat[..., i] ** (1.0 / (i + 1))
            return y

        return _root(points)

    def mirror(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Mirror: keep points already in the ordered simplex $T_d$ fixed and
        reflect every other point into it.

        Based on [1], Sec. 2.3, and [2], Sec. 4.3.3. This implementation covers
        the explicit formulas for $d \in \{1, 2, 3\}$. Mirror is fast but
        discontinuous, and (unlike Root or Shift)
        folds half of any point set that is symmetric about its center (e.g. a
        lattice) on top of the other half.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d),
                $d \in \{1, 2, 3\}$

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Raises:
            NotImplementedError: if the points have dimension greater than 3

        Examples:
            >>> import numpy as np
            >>> transformer = _SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> transformer.mirror(points)
            array([[0.3, 0.7],
                   [0.2, 0.6]])
        """
        y = self._validate_points(points).copy()
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
            "see [1], Sec. 2.3, and [2], Sec. 4.3.3."
        )

    def origami(self, points: np.ndarray, base: int = 2, depth: int = 1) -> np.ndarray:
        r"""
        Transformation Origami: recursively apply Sort within a grid of cubes,
        from the finest scale down to the whole cube.

        Based on [2], Sec. 4.3.4. Choosing a base $b$ and depth $m$ (so the
        finest grid has $M = b^m$ cells per axis), Origami divides the unit
        cube into $M^d$ cells and applies Sort within each; then repeats at
        grid resolutions $M/b, M/b^2, \dots, b, 1$, always operating on the
        current (already partly transformed) point, with the final $N=1$ pass
        equal to a plain global Sort. Origami is discontinuous but, unlike
        Sort, keeps every elementary interval (see [2], Sec. 2.3.5) the same
        size after the transformation. `depth=0` reduces to plain Sort.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)
            base (int): grid subdivisions per level, $b \ge 2$
            depth (int): number of levels above the base grid, $m \ge 0$

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> transformer = _SimplexTransform(dimension=2)
            >>> transformer.origami(np.array([0.9, 0.3]), base=2, depth=1)
            array([[0.4, 0.8]])
        """
        x = self._validate_points(points)
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

        def _origami(flat):
            y = flat.copy()
            for n in (b ** k for k in range(depth, -1, -1)):
                cell = np.floor(n * y)
                frac = n * y - cell
                frac.sort(axis=-1)
                y = (cell + frac) / n
            return y

        return _origami(x)

    def shift(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Shift: push the unit cube into the simplex $A_d$
        (Eq. 4.5), then map $A_d$ onto the simplex $T_d$ used elsewhere in
        this class (Eq. 4.6).

        Based on [2], Sec. 4.3.6. Sorting the input ascending, then for
        $k = 1, \dots, d-1$ subtracting $(d-k)/(d-k+1)$ times the gap between
        sorted coordinates $k-1$ and $k$ from every sorted coordinate at or
        after position $k$, maps the unit cube onto
        $A_d = \{x \in I^d : \sum x < 1\}$; unsorting and taking the
        cumulative sum (Eq. 4.6) then maps $A_d$ onto $T_d$. Shift is
        continuous for any dimension $d$, with no free parameters, and (per
        the elementary-interval comparison in [2], Sec. 4.4) keeps elementary
        intervals more compact than Root does.

        Args:
            points (np.ndarray): Points in the unit cube, shape (..., d)

        Returns:
            np.ndarray: Transformed points in the ordered simplex $T_d$

        Examples:
            >>> import numpy as np
            >>> transformer = _SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> transformer.shift(points)
            array([[0.15, 0.7 ],
                   [0.6 , 0.8 ]])
        """
        points = self._validate_points(points)
        d = points.shape[-1]
        if d == 1:
            return points.copy()

        def _shift(flat):
            order = np.argsort(flat, axis=-1)
            x = np.take_along_axis(flat, order, axis=-1)
            for j in range(d - 1):
                prev = x[..., j - 1] if j >= 1 else 0.0
                gap = x[..., j] - prev
                coeff = (d - j - 1) / (d - j)
                x[..., j:] -= (coeff * gap)[..., None]
            inverse_order = np.argsort(order, axis=-1)
            a = np.take_along_axis(x, inverse_order, axis=-1)
            return np.cumsum(a, axis=-1)

        return _shift(points)
