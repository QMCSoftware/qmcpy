from .abstract_discrete_distribution import AbstractIIDDiscreteDistribution
from ..util import ParameterError, ParameterWarning
import numpy as np
import warnings


class IIDStdUniform(AbstractIIDDiscreteDistribution):
    r"""IID standard uniform points, a wrapper around
    [`numpy.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html).

    Notes:
        - Unlike low discrepancy sequence, calling an `IIDStdUniform` instance gives new samples every time,
            e.g., running the first doctest below with `dd = Lattice(dimension=2)` would give the same 4 points in both calls,
            but since we are using an `IIDStdUniform` instance it gives different points every call.

    Examples:
        >>> discrete_distrib = IIDStdUniform(dimension=2,seed=7)
        >>> discrete_distrib(4)
        array([[0.04386058, 0.58727432],
               [0.3691824 , 0.65212985],
               [0.69669968, 0.10605352],
               [0.63025643, 0.13630282]])
        >>> discrete_distrib(4) # gives new samples every time
        array([[0.5968363 , 0.0576251 ],
               [0.2028797 , 0.22909681],
               [0.1366783 , 0.75220658],
               [0.84501765, 0.56269008]])
        >>> discrete_distrib
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               2^(1)
            replications    1
            entropy         7

        Replications (implemented for API consistency)


        >>> x = IIDStdUniform(dimension=3,replications=2,seed=7)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.65212985, 0.69669968, 0.10605352],
                [0.63025643, 0.13630282, 0.5968363 ],
                [0.0576251 , 0.2028797 , 0.22909681]],
        <BLANKLINE>
               [[0.1366783 , 0.75220658, 0.84501765],
                [0.56269008, 0.04826852, 0.71308655],
                [0.80983568, 0.85383675, 0.80475135],
                [0.6171181 , 0.1239209 , 0.16809479]]])
    """

    def __init__(self, dimension: int = 1, replications=None, seed=None) -> None:
        r"""
        Args:
            dimension (int): Dimension of the samples.
            replications (Union[None, int]): Number of randomizations. This is
                implemented only for API consistency. Equivalent to reshaping
                samples.
            seed (Union[None, int, np.random.SeedSequence]): Seed the random
                number generator for reproducibility.
        """
        super(IIDStdUniform, self).__init__(
            int(dimension), replications, seed, d_limit=np.inf, n_limit=np.inf
        )
        if not (self.dvec == np.arange(self.d)).all():
            warnings.warn("IIDStdUniform does not accomodate dvec", ParameterWarning)

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if n_min > 0 and warn:
            warnings.warn(
                "For IIDStdUniform setting n_min>0 makes no difference as new samples are generated every call regardless, e.g., calling gen_samples(n_min=0,n_max=4) twice will give different samples.",
                ParameterWarning,
            )
        if return_binary:
            raise ParameterError("IIDStdUniform does not support return_binary=True")
        x = self.rng.uniform(size=(self.replications, int(n_max - n_min), self.d))
        return x

    def _spawn(self, child_seed, dimension):
        return IIDStdUniform(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
        )
