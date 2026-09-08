from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from ..util import ParameterError
import numpy as np


class Genz(AbstractIntegrand):
    r"""Genz function following the [`DAKOTA`
    implementation](https://snl-dakota.github.io/docs/6.17.0/users/usingdakota/examples/additionalexamples.html?highlight=genz#genz-functions).

    $$g_\mathrm{oscillatory}(\boldsymbol{t}) = \cos\left(-\sum_{j=1}^d c_j
    t_j\right)$$

    or

    $$g_\mathrm{corner-peak}(\boldsymbol{t}) = \left(1+\sum_{j=1}^d c_j
    t_j\right)^{-(d+1)}$$

    where

    $$\boldsymbol{T} \sim \mathcal{U}[0,1]^d$$

    and the coefficients $\boldsymbol{c}$ are have three kinds

    $$c_k^{(1)} = \frac{k-1/2}{d}, \qquad c_k^{(2)} = \frac{1}{k^2}, \qquad
    c_k^{(3)} = \exp\left(\frac{k \log(10^{-8})}{d}\right), \qquad
    k=1,\dots,d.$$

    Examples:
        >>> for kind_func in ['OSCILLATORY','CORNER PEAK']:
        ...     for kind_coeff in [1,2,3]:
        ...         integrand = Genz(DigitalNetB2(2,seed=7),kind_func=kind_func,kind_coeff=kind_coeff)
        ...         y = integrand(2**14)
        ...         mu_hat = y.mean()
        ...         print('%-15s %-3d %.3f'%(kind_func,kind_coeff,mu_hat))
        OSCILLATORY     1   -0.351
        OSCILLATORY     2   -0.329
        OSCILLATORY     3   -0.217
        CORNER PEAK     1   0.713
        CORNER PEAK     2   0.714
        CORNER PEAK     3   0.720

        With independent replications

        >>> integrand = Genz(DigitalNetB2(2,seed=7,replications=2**4),kind_func="CORNER PEAK",kind_coeff=3)
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean())
        0.7200
    """

    def __init__(self, sampler, kind_func: str = "OSCILLATORY", kind_coeff: int = 1) -> None:
        """Initialize a Genz integrand.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            kind_func (str): Either `'OSCILLATORY'` or `'CORNER PEAK'`
            kind_coeff (int): 1, 2, or 3 for choice of coefficients
        """
        self.kind_func = (
            str(kind_func).upper().strip().replace("_", " ").replace("-", " ")
        )
        self.kind_coeff = kind_coeff
        if (self.kind_func not in ["OSCILLATORY", "CORNER PEAK"]) or (
            self.kind_coeff not in [1, 2, 3]
        ):
            raise ParameterError(
                """
                Genz expects 
                    kind_func in ['OSCILLATORY','CORNER PEAK'] and 
                    kind_coeffs in [1,2,3]"""
            )
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler)
        self.d = self.true_measure.d
        if self.kind_coeff == 1:
            self.c = (np.arange(1, self.d + 1) - 0.5) / self.d
        elif self.kind_coeff == 2:
            self.c = 1 / np.arange(1, self.d + 1) ** 2
        elif self.kind_coeff == 3:
            self.c = np.exp(np.arange(1, self.d + 1) * np.log(10 ** (-8)) / self.d)
        if self.kind_func == "OSCILLATORY":
            self.g = self.g_oscillatory
            self.c = 4.5 * self.c / self.c.sum()
        elif self.kind_func == "CORNER PEAK":
            self.g = self.g_corner_peak
            self.c = 0.25 * self.c / self.c.sum()
        self.c = self.c[None, :]
        self.parameters = ["kind_func", "kind_coeff"]
        super(Genz, self).__init__(dimension_indv=(), dimension_comb=(), parallel=False)

    def g_oscillatory(self, t):
        r"""Evaluate the oscillatory Genz function.

        Args:
            t (np.ndarray): Points in the unit cube.

        Returns:
            np.ndarray: $\cos(-c \cdot t)$ at each point.
        """
        return np.cos(-(self.c * t).sum(-1))

    def g_corner_peak(self, t):
        r"""Evaluate the corner-peak Genz function.

        Args:
            t (np.ndarray): Points in the unit cube.

        Returns:
            np.ndarray: $(1 + c \cdot t)^{-(d+1)}$ at each point.
        """
        return (1 + (self.c * t).sum(-1)) ** (-(self.d + 1))

    def _spawn(self, level, sampler):
        return Genz(
            sampler=sampler, kind_func=self.kind_func, kind_coeff=self.kind_coeff
        )
