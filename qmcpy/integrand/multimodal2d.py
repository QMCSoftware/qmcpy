import numpy as np
from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2


class Multimodal2d(AbstractIntegrand):
    r"""
    Multimodal function in $d=2$ dimensions.

    $$g(\boldsymbol{t}) = (t_0^2+4)(t_1-1)/20-\sin(5t_0/2)-2 \qquad \boldsymbol{T} = (T_0,T_1) \sim \mathcal{U}([-4,7] \times [-3,8]).$$

    Examples:
        >>> integrand = Multimodal2d(DigitalNetB2(2,seed=7))
        >>> y = integrand(2**10)
        >>> print("%.4f"%y.mean())
        -0.7365
        >>> integrand.true_measure  # doctest: +NORMALIZE_WHITESPACE
        Uniform (AbstractTrueMeasure)
            lower_bound     [-4 -3]
            upper_bound     [7 8]
            mean            [1.5 2.5]
            variance        [10.083 10.083]
            standard_deviation [3.175 3.175]
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                             with 2 stored elements (1 diagonals) and shape (2, 2)>
                              Coords Values
                              (0, 0) 10.083333333333334
                              (1, 1) 10.083333333333334

        With independent replications

        >>> integrand = Multimodal2d(DigitalNetB2(2,seed=7,replications=2**4))
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean())
        -0.7366
    """

    def __init__(self, sampler):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
        """
        self.sampler = sampler
        assert self.sampler.d == 2
        self.true_measure = Uniform(
            self.sampler, lower_bound=[-4, -3], upper_bound=[7, 8]
        )
        super(Multimodal2d, self).__init__(
            dimension_indv=(), dimension_comb=(), parallel=False
        )

    def g(self, t):
        t0, t1 = t[..., 0], t[..., 1]
        return (t0**2 + 4) * (t1 - 1) / 20 - np.sin(5 * t0 / 2) - 2

    def _spawn(self, level, sampler):
        return Multimodal2d(sampler=sampler)
