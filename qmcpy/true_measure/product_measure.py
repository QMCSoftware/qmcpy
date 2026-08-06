import numpy as np

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import DimensionError, ParameterError


class ProductMeasure(AbstractTrueMeasure):
    r"""
    Product true measure for independent composition of marginal true measures.

    ``ProductMeasure`` represents an independent product of smaller true
    measures. Each marginal may be one-dimensional or multidimensional. If the
    marginal true measures have dimensions

    d_1, d_2, ..., d_k,

    then the product measure has total dimension

    d = d_1 + d_2 + ... + d_k.

    A single d-dimensional outer sampler is used. Its unit-cube samples are
    split into coordinate blocks, one block for each marginal true measure.
    Each marginal transforms its own block, and the transformed blocks are
    concatenated back together.

    For example, if the marginals are

        marginal 1: 2D Gaussian
        marginal 2: 1D zero-inflated exponential

    then ``ProductMeasure`` uses a 3D sampler and returns samples with three
    coordinates. The first two coordinates come from the Gaussian marginal,
    and the third coordinate comes from the zero-inflated exponential marginal.

    The marginal true measures still have their own samplers because QMCPy's
    current ``AbstractTrueMeasure`` API requires every true measure to be
    constructed with one. A ``DummySampler`` is useful for this construction
    placeholder role. Inside ``ProductMeasure``, marginal samplers are not
    sampled directly when product samples are generated. The marginals provide
    dimension, range, transform, and weight behavior. A future
    samplerless/template true-measure mode may be useful, but that is separate
    from this class.

    Notes
    -----
    Exact product weights are supported for direct marginal true measures. For
    recursively composed marginal measures, sampling is supported through
    QMCPy's recursive transform helper, but exact final-space product weights
    are not currently implemented here.

    Examples
    --------
    Combine two one-dimensional uniform true measures:

    >>> from qmcpy import DigitalNetB2, DummySampler, ProductMeasure, Uniform
    >>> marginals = [
    ...     Uniform(DummySampler(1), lower_bound=0, upper_bound=2),
    ...     Uniform(DummySampler(1), lower_bound=10, upper_bound=12),
    ... ]
    >>> pm = ProductMeasure(sampler=DigitalNetB2(2, seed=9), marginals=marginals)
    >>> x = pm(4)
    >>> x.shape
    (4, 2)
    >>> bool(((0 <= x[:, 0]) & (x[:, 0] <= 2)).all())
    True

    The outer sampler controls replications:

    >>> pm = ProductMeasure(
    ...     sampler=DigitalNetB2(2, seed=9, replications=3),
    ...     marginals=marginals,
    ... )
    >>> pm(4).shape
    (3, 4, 2)

    The ``DummySampler`` marginal samplers are only construction placeholders
    required by the current ``AbstractTrueMeasure`` interface.
    ``ProductMeasure`` samples from its own outer sampler.

    Marginals may have different dimensions:

    >>> import numpy as np
    >>> from qmcpy import Gaussian
    >>> marginals = [
    ...     Gaussian(
    ...         DummySampler(2),
    ...         mean=[0, 0],
    ...         covariance=np.eye(2),
    ...     ),
    ...     Uniform(DummySampler(1), lower_bound=10, upper_bound=12),
    ... ]
    >>> pm = ProductMeasure(sampler=DigitalNetB2(3, seed=12), marginals=marginals)
    >>> pm(4).shape
    (4, 3)
    """

    def __init__(self, sampler, marginals):
        """
        Initialize a product measure from one sampler and several marginals.

        Parameters
        ----------
        sampler : AbstractDiscreteDistribution
            The sampler for the whole product measure. Its dimension must
            equal the sum of the marginal dimensions.

        marginals : list or tuple of AbstractTrueMeasure
            Independent true measures to place side by side. A marginal may
            itself be multidimensional.

        Why one sampler?
        ----------------
        The product measure should be driven by one total-dimensional QMC
        point set. We do not generate separate QMC samples from each marginal.
        Instead, one sample u in [0,1]^d is split into blocks:

            u = (u_marginal_1, u_marginal_2, ..., u_marginal_k).

        This preserves the intended total-dimensional QMC construction.
        """
        if not isinstance(marginals, (list, tuple)) or len(marginals) == 0:
            raise ParameterError("ProductMeasure requires a nonempty list of marginals.")

        if not all(isinstance(marginal, AbstractTrueMeasure) for marginal in marginals):
            raise ParameterError(
                "Each ProductMeasure marginal must be an AbstractTrueMeasure instance."
            )

        if not isinstance(sampler, AbstractDiscreteDistribution):
            raise ParameterError(
                "ProductMeasure sampler must be an AbstractDiscreteDistribution."
            )

        self.parameters = ["marginals"]
        # ProductMeasure uses only the sampler passed directly to ProductMeasure
        # to generate product samples.
        #
        # Marginal true measures also contain samplers because QMCPy's current
        # AbstractTrueMeasure interface requires true measures to be constructed
        # with an attached discrete distribution. Inside ProductMeasure, those
        # marginal samplers are not sampled. The marginals are used for their
        # dimension, range, transform, and weight behavior.
        self.marginals = list(marginals)
        for marginal in self.marginals:
            target_dim = getattr(marginal, "target_dim", marginal.d)
            if target_dim != marginal.d:
                raise DimensionError(
                    "ProductMeasure marginals must be dimension-preserving "
                    "block transforms. Marginal target dimension "
                    f"{target_dim} does not match sampler dimension {marginal.d}."
                )

        self.marginal_dimensions = np.array(
            [marginal.d for marginal in self.marginals], dtype=int
        )
        self._split_indices = np.cumsum(self.marginal_dimensions)[:-1]

        self._total_marginal_dimension = int(self.marginal_dimensions.sum())
        if sampler.d != self._total_marginal_dimension:
            raise DimensionError(
                "ProductMeasure sampler dimension must equal the sum of marginal "
                f"dimensions ({sampler.d} != {self._total_marginal_dimension})."
            )

        self.domain = np.array([[0.0, 1.0]])
        self._parse_sampler(sampler)

        self.range = np.vstack(
            [
                self._expand_bounds(marginal.range, marginal.d, "range")
                for marginal in self.marginals
            ]
        )

        super(ProductMeasure, self).__init__()

    @staticmethod
    def _expand_bounds(bounds, dimension, name):
        """
        Expand a marginal's bounds so they have one row per output coordinate.

        Some true measures store bounds as shape (1, 2), meaning the same
        bound applies to all coordinates. Others store bounds as shape
        (dimension, 2), meaning each coordinate has its own bound.

        ProductMeasure needs all marginal ranges stacked together, so every
        marginal range must be represented as shape (dimension, 2).
        """
        bounds = np.asarray(bounds)

        if bounds.shape == (1, 2):
            return np.tile(bounds, (dimension, 1))

        if bounds.shape == (dimension, 2):
            return bounds

        raise DimensionError(
            f"Marginal true measure {name} must have shape (1, 2) or ({dimension}, 2)."
        )

    @property
    def _has_recursive_marginal(self):
        """
        Check whether any marginal is itself recursively composed.

        In QMCPy, a true measure can sometimes be built on top of another true
        measure. Sampling can still be handled by the recursive transform
        helper, but exact product weights in the final transformed space are
        more delicate. For now, ProductMeasure only computes exact weights
        when all marginals are direct true measures.
        """
        return any(marginal.transform != marginal for marginal in self.marginals)

    def _split_blocks(self, x):
        """
        Split an input array into marginal coordinate blocks.

        The split always happens along the final axis, so this works for both
        ordinary samples with shape (n, d) and replicated samples with shape
        (r, n, d).
        """
        x = np.asarray(x, dtype=float)

        if x.shape[-1] != self.d:
            raise DimensionError(
                f"ProductMeasure expected last axis {self.d}, got {x.shape[-1]}."
            )

        return np.split(x, self._split_indices, axis=-1)

    def _transform(self, x):
        """
        Transform unit-cube samples into product-measure samples.

        Steps
        -----
        1. Split the full unit-cube sample into marginal blocks.
        2. Send each block to the matching marginal true measure.
        3. Concatenate the transformed marginal outputs.

        This implements

            T(u) = (T_1(u_1), T_2(u_2), ..., T_k(u_k)),

        where each marginal T_j acts only on its own coordinate block.
        """
        blocks = self._split_blocks(x)

        transformed_blocks = [
            marginal._jacobian_transform_r(block, return_weights=False)
            for marginal, block in zip(self.marginals, blocks)
        ]

        return np.concatenate(transformed_blocks, axis=-1)

    def _weight(self, x):
        """
        Compute the product density/weight for independent marginals.

        For independent components, the joint weight is the product of the
        marginal weights:

            w(x) = w_1(x_1) * w_2(x_2) * ... * w_k(x_k).

        This method supports direct marginal true measures. Recursive
        marginals are blocked for now because their final-space weights need
        more careful handling.
        """
        if self._has_recursive_marginal:
            raise ParameterError(
                "ProductMeasure exact weights are currently supported only for "
                "direct marginal true measures."
            )

        blocks = self._split_blocks(x)
        weight = np.ones(np.asarray(x).shape[:-1], dtype=float)

        for marginal, block in zip(self.marginals, blocks):
            weight *= marginal._weight(block)

        return weight

    def _spawn(self, sampler, dimension):
        """
        Spawn a new ProductMeasure with a new outer sampler.

        QMCPy's spawn mechanism creates new randomized copies of a sampler or
        true measure. ProductMeasure preserves the same marginal structure and
        replaces only the outer product sampler.
        """
        if dimension != self.d:
            raise DimensionError(
                "ProductMeasure spawning currently preserves the marginal dimensions."
            )

        return ProductMeasure(sampler=sampler, marginals=self.marginals)
