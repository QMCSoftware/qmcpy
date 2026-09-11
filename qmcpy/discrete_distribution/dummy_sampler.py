from typing import Union
import numpy as np
from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..util import ParameterError


class DummySampler(AbstractLDDiscreteDistribution):
    r"""Placeholder discrete distribution for constructing true-measure
    marginals.

    ``DummySampler`` is useful when a true measure is needed only for its
    dimension, transform, range, and weight behavior. QMCPy's current
    ``AbstractTrueMeasure`` interface requires each true measure to be
    constructed with an attached sampler, but ``ProductMeasure`` samples only
    from its own outer sampler.

    Direct calls to ``DummySampler`` raise an error because the sampler is only
    a construction placeholder and cannot generate meaningful QMC points.

    Examples:
    >>> from qmcpy import DummySampler
    >>> sampler = DummySampler(2)
    >>> sampler.d
    2
    >>> sampler.replications
    1
    >>> sampler(4)
    Traceback (most recent call last):
        ...
    qmcpy.util.exceptions_warnings.ParameterError: DummySampler is only a construction placeholder for ProductMeasure child true measures and cannot generate samples.
    """

    def __init__(self, dimension: int = 1, replications: Union[None, int] = None, seed: Union[None, int, np.random.SeedSequence] = None, warn: bool = True) -> None:
        """Initialize a DummySampler discrete distribution.

        Args:
            dimension (int): Dimension of the placeholder sampler.
            replications (Union[None, int]): Number of independent randomizations, kept
                for API consistency with the other discrete distributions.
            seed (Union[None, int, np.random.SeedSequence]): Unused; kept for
                API consistency with the other discrete distributions.
            warn (bool): Unused; kept for API consistency with the other
                discrete distributions.
        """
        # Keep the same constructor as other discrete distributions.
        del warn

        # DummySampler has no extra parameters.
        self.parameters = []

        # True measures expect unit-cube input.
        self.mimics = "StdUniform"

        # Initialize the common discrete distribution settings.
        super(DummySampler, self).__init__(
            dimension,
            replications,
            seed,
            d_limit=10_000,
            n_limit=2**32,
        )

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        raise ParameterError(
            "DummySampler is only a construction placeholder for ProductMeasure "
            "child true measures and cannot generate samples."
        )

    def _spawn(self, child_seed, dimension):
        # Create a new DummySampler with the given dimension and seed.
        # Preserve the current replication setting.
        return DummySampler(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            warn=False,
        )
