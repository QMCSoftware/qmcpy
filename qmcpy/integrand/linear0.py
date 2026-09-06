from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform


class Linear0(AbstractIntegrand):
    r"""Linear Function with analytic mean $0$.

    $$g(\boldsymbol{t}) = \sum_{j=1}^d t_j \qquad \boldsymbol{T} \sim
    \mathcal{U}[0,1]^d.$$

    Examples:
        >>> integrand = Linear0(DigitalNetB2(100,seed=7))
        >>> y = integrand(2**10)
        >>> print("%.4e"%y.mean())
        6.0560e-05

        With independent replications

        >>> integrand = Linear0(DigitalNetB2(100,seed=7,replications=2**4))
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4e"%muhats.mean())
        -9.8203e-05
    """

    def __init__(self, sampler) -> None:
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
        """
        self.sampler = sampler
        self.true_measure = Uniform(self.sampler, lower_bound=-0.5, upper_bound=0.5)
        super(Linear0, self).__init__(
            dimension_indv=(), dimension_comb=(), parallel=False
        )

    def g(self, t):
        y = t.sum(-1)
        return y

    def _spawn(self, level, sampler):
        return Linear0(sampler=sampler)
