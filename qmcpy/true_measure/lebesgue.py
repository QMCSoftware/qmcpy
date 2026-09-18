from .abstract_true_measure import AbstractTrueMeasure
from .uniform import Uniform
from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
import numpy as np


class Lebesgue(AbstractTrueMeasure):
    r"""
    Lebesgue measure as described in [https://en.wikipedia.org/wiki/Lebesgue_measure](https://en.wikipedia.org/wiki/Lebesgue_measure).

    ``Lebesgue`` supplies the constant target weight one. Its sampler is a true
    measure that defines the integration region and proposal geometry; it is
    not treated as an ordinary deterministic transport. Use the resulting
    target with explicit ``ImportanceSampling``.

    Examples:
        >>> from qmcpy import DigitalNetB2, ImportanceSampling, Lebesgue, Uniform
        >>> proposal = Uniform(
        ...     DigitalNetB2(1,seed=7),
        ...     lower_bound=1,
        ...     upper_bound=3,
        ... )
        >>> target = Lebesgue(proposal)
        >>> importance_sampling = ImportanceSampling(
        ...     target=target,
        ...     proposal=proposal,
        ... )
        >>> samples, weights = importance_sampling.gen_samples(
        ...     4,
        ...     return_weights=True,
        ... )
        >>> samples.shape, weights.shape
        ((4, 1), (4,))
        >>> bool(np.all(weights == 2))
        True
    """

    def __init__(self, sampler: AbstractTrueMeasure) -> None:
        r"""Initialize a Lebesgue true measure.

        Args:
            sampler (AbstractTrueMeasure): Measure defining the integration region and proposal geometry for the constant target weight one.
        """
        self.parameters = []
        if not isinstance(sampler, AbstractTrueMeasure):
            raise ParameterError(
                "Lebesgue sampler must be an AbstractTrueMeasure defining its integration region."
            )
        self.domain = (
            sampler.range
        )  # hack to make sure Lebesgue is compatible with any transform
        self.range = sampler.range
        self._parse_sampler(sampler)
        super(Lebesgue, self).__init__()

    def _weight(self, x):
        return np.ones(x.shape[:-1], dtype=float)

    def _spawn(self, sampler, dimension):
        return Lebesgue(sampler)
