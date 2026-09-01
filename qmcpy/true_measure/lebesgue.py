from .abstract_true_measure import AbstractTrueMeasure
from .uniform import Uniform
from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
import numpy as np


class Lebesgue(AbstractTrueMeasure):
    r"""
    Lebesgue measure as described in [https://en.wikipedia.org/wiki/Lebesgue_measure](https://en.wikipedia.org/wiki/Lebesgue_measure).

    Examples:
        >>> Lebesgue(Gaussian(DigitalNetB2(2,seed=7)))
        Lebesgue (AbstractTrueMeasure)
            transform       Gaussian (AbstractTrueMeasure)
                                mean            [0. 0.]
                                variance        [1. 1.]
                                standard_deviation [1. 1.]
                                covariance      [[1. 0.]
                                                 [0. 1.]]
                                decomp_type     PCA
        >>> Lebesgue(Uniform(DigitalNetB2(2,seed=7)))  # doctest: +NORMALIZE_WHITESPACE
        Lebesgue (AbstractTrueMeasure)
            transform       Uniform (AbstractTrueMeasure)
                                lower_bound     0
                                upper_bound     1
                                mean            [0.5 0.5]
                                variance        [0.083 0.083]
                                standard_deviation [0.289 0.289]
                                covariance      <DIAgonal sparse matrix of dtype 'float64'
                                                 with 2 stored elements (1 diagonals) and shape (2, 2)>
                                                  Coords Values
                                                  (0, 0) 0.08333333333333333
                                                  (1, 1) 0.08333333333333333
    """

    def __init__(self, sampler):
        r"""
        Args:
            sampler (AbstractTrueMeasure): A true measure by which to compose a transform.
        """
        self.parameters = []
        if not isinstance(sampler, AbstractTrueMeasure):
            raise ParameterError(
                "Lebesgue sampler must be an AbstractTrueMeasure by which to transform samples."
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
