import numpy as np
from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2


class Sin1d(AbstractIntegrand):
    r"""
    Sine function in $d=1$ dimension.

    $$g(t) = \sin(t), \qquad t \sim \mathcal{U}[0,2\pi k]$$

    Examples:
        >>> integrand = Sin1d(DigitalNetB2(1,seed=7),k=1)
        >>> y = integrand(2**10)
        >>> print("%.4e"%y.mean())
        -1.3582e-10
        >>> integrand.true_measure  # doctest: +NORMALIZE_WHITESPACE
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     6.283
            mean            3.142
            variance        3.290
            standard_deviation 1.814
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                             with 1 stored elements (1 diagonals) and shape (1, 1)>
                              Coords Values
                              (0, 0) 3.289868133696453

        With independent replications

        >>> integrand = Sin1d(DigitalNetB2(1,seed=7,replications=2**4),k=1)
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4e"%muhats.mean())
        7.0800e-04
    """

    def __init__(self, sampler, k=1):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            k (float): The true measure will be uniform between $0$ and $2 \pi k$.
        """
        self.sampler = sampler
        self.k = k
        assert self.sampler.d == 1
        self.true_measure = Uniform(
            self.sampler, lower_bound=0, upper_bound=2 * self.k * np.pi
        )
        super(Sin1d, self).__init__(
            dimension_indv=(), dimension_comb=(), parallel=False
        )

    def g(self, t):
        return np.sin(t[..., 0])

    def _spawn(self, level, sampler):
        return Sin1d(sampler=sampler, k=self.k)
