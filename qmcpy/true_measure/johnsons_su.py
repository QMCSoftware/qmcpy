from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from typing import Union
from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from scipy.stats import norm


class JohnsonsSU(AbstractTrueMeasure):
    r"""Johnson's $S_U$-distribution with independent marginals as described
    in
    [https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution](https://en.wikipedia.org/wiki/Johnson%27s_SU-distribution).

    Examples:
        >>> true_measure = JohnsonsSU(DigitalNetB2(2,seed=7),gamma=1,xi=2,delta=3,lam=4)
        >>> true_measure(4)
        array([[ 1.44849599,  2.49715741],
               [-0.83646172,  0.38970902],
               [ 3.67068094, -2.33911196],
               [ 0.38940887,  0.84843818]])
        >>> true_measure
        JohnsonsSU (AbstractTrueMeasure)
            gamma           1
            xi              2^(1)
            delta           3
            lam             2^(2)

        With independent replications

        >>> x = JohnsonsSU(DigitalNetB2(3,seed=7,replications=2),gamma=1,xi=2,delta=3,lam=4)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[-0.36735335, -0.71750135,  1.55387818],
                [ 1.29233112,  1.21788962,  0.3870404 ],
                [ 0.57599197,  1.78008445, -1.53756327],
                [ 2.50112084, -0.14204369,  1.67333839]],
        <BLANKLINE>
               [[ 0.45920696,  2.10110361,  0.66122546],
                [ 0.76973983, -2.12724026,  0.02868419],
                [-0.43948474, -0.1525475 , -1.71041918],
                [ 1.57765245,  1.00275   ,  1.64972468]]])
    """

    def __init__(self, sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure], gamma: Union[float, np.ndarray] = 1, xi: Union[float, np.ndarray] = 1, delta: Union[float, np.ndarray] = 2, lam: Union[float, np.ndarray] = 2) -> None:
        r"""Initialize a JohnsonsSU true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            gamma (Union[float, np.ndarray]): First parameter $\gamma$.
            xi (Union[float, np.ndarray]): Second parameter $\xi$.
            delta (Union[float, np.ndarray]): Third parameter $\delta > 0$.
            lam (Union[float, np.ndarray]): Fourth parameter $\lambda > 0$.
        """
        self.parameters = ["gamma", "xi", "delta", "lam"]
        self.domain = np.array([[0, 1]])
        self.range = np.array([[-np.inf, np.inf]])
        self._parse_sampler(sampler)
        self.gamma = gamma
        self.xi = xi
        self.delta = delta
        self.lam = lam
        self._gamma = np.array(gamma)
        if self._gamma.size == 1:
            self._gamma = self._gamma.item() * np.ones(self.d)
        self._xi = np.array(xi)
        if self._xi.size == 1:
            self._xi = self._xi.item() * np.ones(self.d)
        self._delta = np.array(delta)
        if self._delta.size == 1:
            self._delta = self._delta.item() * np.ones(self.d)
        self._lam = np.array(lam)
        if self._lam.size == 1:
            self._lam = self._lam.item() * np.ones(self.d)
        if not (
            self._gamma.shape == (self.d,)
            and self._xi.shape == (self.d,)
            and self._delta.shape == (self.d,)
            and self._lam.shape == (self.d,)
        ):
            raise DimensionError(
                "all Johnson's S_U parameters be scalar or have length equal to dimension."
            )
        if not ((self._delta > 0).all() and (self._lam > 0).all()):
            raise ParameterError("delta and lam must be all be positive")
        super(JohnsonsSU, self).__init__()
        if not (
            self._gamma.shape == (self.d,)
            and self._xi.shape == (self.d,)
            and self._delta.shape == (self.d,)
            and self._lam.shape == (self.d,)
        ):
            raise AssertionError

    def _transform(self, x):
        return self._lam * np.sinh((norm.ppf(x) - self._gamma) / self._delta) + self._xi

    def _weight(self, x):
        term1 = (x - self._xi) / self._lam
        term2 = (
            self._delta / (self._lam * np.sqrt(2 * np.pi)) * 1 / np.sqrt(1 + term1**2)
        )
        term3 = np.exp(-1 / 2 * (self._gamma + self._delta * np.arcsinh(term1)) ** 2)
        return np.prod(term2 * term3, -1)

    def _spawn(self, sampler, dimension):
        if dimension == self.d:  # don't do anything if the dimension doesn't change
            spawn = JohnsonsSU(
                sampler,
                gamma=self._gamma,
                xi=self._xi,
                delta=self._delta,
                lam=self._lam,
            )
        else:
            gamma = self._gamma[0]
            xi = self._xi[0]
            delta = self._delta[0]
            lam = self._lam[0]
            if not (
                all(self._gamma == gamma)
                and all(self._xi == xi)
                and all(self._delta == delta)
                and all(self._lam == lam)
            ):
                raise DimensionError(
                    """
                    In order to spawn a JohnsonsSU measure
                    gamma must all be the same and 
                    xi must all be the same and 
                    delta must all be the same and 
                    lam (lambda) must all be the same."""
                )
            spawn = JohnsonsSU(sampler, gamma=gamma, xi=xi, delta=delta, lam=lam)
        return spawn
