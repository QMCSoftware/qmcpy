from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian
import numpy as np
from scipy.special import gamma


class Keister(AbstractIntegrand):
    r"""
    Keister function from [1].

    $$f(\boldsymbol{t}) = \pi^{d/2} \cos(\lVert \boldsymbol{t} \rVert_2) \qquad \boldsymbol{T} \sim \mathcal{N}(\boldsymbol{0},\mathsf{I}/2).$$

    Examples:
        >>> integrand = Keister(DigitalNetB2(2,seed=7))
        >>> y = integrand(2**10)
        >>> print("%.4f"%y.mean())
        1.8080
        >>> integrand.true_measure
        Gaussian (AbstractTrueMeasure)
            mean            [0. 0.]
            variance        [0.5 0.5]
            standard_deviation [0.707 0.707]
            covariance      [[0.5 0. ]
                             [0.  0.5]]
            decomp_type     PCA

        With independent replications

        >>> integrand = Keister(DigitalNetB2(2,seed=7,replications=2**4))
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean())
        1.8024

    **References:**

    1.  B. D. Keister.
        Multidimensional Quadrature Algorithms.
        Computers in Physics, 10, pp. 119-122, 1996.
    """

    def __init__(self, sampler):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
        """
        self.sampler = sampler
        self.true_measure = Gaussian(self.sampler, mean=0, covariance=1 / 2)
        super(Keister, self).__init__(
            dimension_indv=(), dimension_comb=(), parallel=False
        )

    def g(self, t):
        d = t.shape[-1]
        norm = np.linalg.norm(t, axis=-1)
        k = np.pi ** (d / 2) * np.cos(norm)
        return k

    def _spawn(self, level, sampler):
        return Keister(sampler=sampler)

    @classmethod
    def get_exact_value(self, d):
        """
        Compute the exact analytic value of the Keister integral with dimension $d$.

        Args:
            d (int): Dimension.

        Returns:
            mean (float): Exact value of the integral.
        """
        cosinteg = np.zeros(shape=(d))
        cosinteg[0] = np.sqrt(np.pi) / (2 * np.exp(1 / 4))
        sininteg = np.zeros(shape=(d))
        sininteg[0] = 4.244363835020225e-01
        if d > 1:
            cosinteg[1] = (1 - sininteg[0]) / 2
            sininteg[1] = cosinteg[0] / 2
        for j in range(2, d):
            cosinteg[j] = ((j - 1) * cosinteg[j - 2] - sininteg[j - 1]) / 2
            sininteg[j] = ((j - 1) * sininteg[j - 2] + cosinteg[j - 1]) / 2
        I = (2 * (np.pi ** (d / 2)) / gamma(d / 2)) * cosinteg[d - 1]
        return I

    def exact_integ(self, *args, **kwargs):
        return self.get_exact_value(*args, **kwargs)
