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
        >>> samples
        array([[1.73585536],
               [2.6197787 ],
               [1.20363334],
               [2.08950073]])
        >>> weights
        array([2., 2., 2., 2.])
        >>> samples.shape, weights.shape
        ((4, 1), (4,))
        >>> bool(np.all(weights == 2))
        True

        The samples follow the proposal. The weights equal the interval length,
        converting its uniform probability density to Lebesgue weight one.
        With independent replications:

        >>> proposal_rep = Uniform(
        ...     DigitalNetB2(1, seed=7, replications=2),
        ...     lower_bound=1, upper_bound=3,
        ... )
        >>> target_rep = Lebesgue(proposal_rep)
        >>> importance_sampling_rep = ImportanceSampling(target_rep, proposal_rep)
        >>> samples_rep, weights_rep = importance_sampling_rep.gen_samples(
        ...     4, return_weights=True,
        ... )
        >>> samples_rep
        array([[[2.44324713],
                [1.32691107],
                [2.97352511],
                [1.8591331 ]],
        <BLANKLINE>
               [[2.82991   ],
                [1.85929712],
                [2.11752684],
                [1.06872767]]])
        >>> weights_rep
        array([[2., 2., 2., 2.],
               [2., 2., 2., 2.]])
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

    @property
    def effective_range(self):
        """Certified integration region supplied by the wrapped measure."""
        effective_range = self.transform.effective_range
        if effective_range is None:
            return None
        return self._read_only_array(effective_range)

    def _weight(self, x):
        return np.ones(x.shape[:-1], dtype=float)

    def _spawn(self, sampler, dimension):
        return Lebesgue(sampler)
