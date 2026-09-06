from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
import numpy as np


class BoxIntegral(AbstractIntegrand):
    r"""Box integral from [1], see also

    $$B_s(\boldsymbol{t}) = \left(\sum_{j=1}^d t_j^2 \right)^{s/2}, \qquad
    \boldsymbol{T} \sim \mathcal{U}[0,1]^d.$$

    Examples:
        Scalar `s`

        >>> integrand = BoxIntegral(DigitalNetB2(2,seed=7),s=7)
        >>> y = integrand(2**10)
        >>> y.shape
        (1024,)
        >>> print("%.4f"%y.mean(0))
        0.7519

        With independent replications

        >>> integrand = BoxIntegral(DigitalNetB2(2,seed=7,replications=2**4),s=7)
        >>> y = integrand(2**10)
        >>> y.shape
        (16, 1024)
        >>> muhats = y.mean(1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean(0))
        0.7518

        Array `s`

        >>> integrand = BoxIntegral(DigitalNetB2(5,seed=7),s=np.arange(6).reshape((2,3)))
        >>> y = integrand(2**10)
        >>> y.shape
        (2, 3, 1024)
        >>> y.mean(-1)
        array([[1.        , 1.26234461, 1.66666661],
               [2.28201516, 3.22195096, 4.67188113]])

        With independent replications

        >>> integrand = BoxIntegral(DigitalNetB2(2,seed=7,replications=2**4),s=np.arange(6).reshape((2,3)))
        >>> y = integrand(2**10)
        >>> y.shape
        (2, 3, 16, 1024)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (2, 3, 16)
        >>> muhats.mean(-1)
        array([[1.        , 0.76519118, 0.66666666],
               [0.62718785, 0.62224086, 0.64273341]])

    **References: **

    1.  D.H. Bailey, J.M. Borwein, R.E. Crandall, Box integrals.
        Journal of Computational and Applied Mathematics, Volume 206, Issue 1, 2007, Pages 196-208, ISSN 0377-0427.
        [https://doi.org/10.1016/j.cam.2006.06.010](https://doi.org/10.1016/j.cam.2006.06.010).
        [https://www.sciencedirect.com/science/article/pii/S0377042706004250](https://www.sciencedirect.com/science/article/pii/S0377042706004250).
        [https://www.davidhbailey.com/dhbpapers/boxintegrals.pdf](https://www.davidhbailey.com/dhbpapers/boxintegrals.pdf)
    """

    def __init__(self, sampler, s=1):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            s (Union[float, np.ndarray]): `s` parameter or parameters. The
                output shape of `g` is the shape of `s`.
        """
        self.parameters = ["s"]
        self.s = np.array(s)
        if not (self.s.size > 0):
            raise AssertionError
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler)
        self.s_over_2 = self.s / 2
        super(BoxIntegral, self).__init__(
            dimension_indv=self.s.shape, dimension_comb=self.s.shape, parallel=False
        )

    def g(self, t, **kwargs):
        sum_squares = (t**2).sum(-1)
        y = sum_squares ** self.s_over_2[(...,) + (None,) * sum_squares.ndim]
        return y

    def _spawn(self, level, sampler):
        return BoxIntegral(sampler=sampler, s=self.s)
