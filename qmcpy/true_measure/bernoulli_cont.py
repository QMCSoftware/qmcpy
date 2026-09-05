from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError
from ..discrete_distribution import DigitalNetB2
import numpy as np


class BernoulliCont(AbstractTrueMeasure):
    r"""
    Continuous Bernoulli distribution with independent marginals as described in [https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution](https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution).

    Examples:
        >>> true_measure = BernoulliCont(DigitalNetB2(2,seed=7),lam=.2)
        >>> true_measure(4)
        array([[0.56205914, 0.83607872],
               [0.09433983, 0.28057299],
               [0.97190779, 0.01883497],
               [0.28050753, 0.39178506]])
        >>> true_measure
        BernoulliCont (AbstractTrueMeasure)
            lam             0.200

        With independent replications

        >>> x = BernoulliCont(DigitalNetB2(3,seed=7,replications=2),lam=[.25,.5,.75])(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.16343492, 0.1821862 , 0.83209443],
                [0.55140696, 0.66169442, 0.56381501],
                [0.35229402, 0.79818233, 0.13825119],
                [0.85773226, 0.29520621, 0.85203898]],
        <BLANKLINE>
               [[0.32359293, 0.85899604, 0.63591945],
                [0.40278251, 0.04353443, 0.46749989],
                [0.1530438 , 0.29281506, 0.116725  ],
                [0.6345258 , 0.60241448, 0.84822692]]])
    """

    def __init__(self, sampler, lam=1 / 2):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            lam (Union[float, np.ndarray]): Vector of shape parameters, each in $(0,1)$.
        """
        self.parameters = ["lam"]
        self.domain = np.array([[0, 1]])
        self.range = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self.lam = lam
        self.l = np.array(lam)
        if self.l.size == 1:
            self.l = self.l.item() * np.ones(self.d)
        if not (
            self.l.shape == (self.d,) and (0 <= self.l).all() and (self.l <= 1).all()
        ):
            raise DimensionError(
                "lam must be scalar or have length equal to dimension and must be in (0,1)."
            )
        super(BernoulliCont, self).__init__()

    def _transform(self, x):
        tf = np.zeros(x.shape, dtype=float)
        for j in range(self.d):
            tf[..., j] = (
                x[..., j]
                if self.l[j] == 1 / 2
                else np.log(
                    ((2 * self.l[j] - 1) * x[..., j] - self.l[j] + 1) / (1 - self.l[j])
                )
                / np.log(self.l[j] / (1 - self.l[j]))
            )
        return tf

    def _weight(self, x):
        in_support = np.all((0 <= x) & (x <= 1), axis=-1)
        w = np.zeros(x.shape, dtype=float)
        for j in range(self.d):
            C = (
                2
                if self.l[j] == 1 / 2
                else 2 * np.arctanh(1 - 2 * self.l[j]) / (1 - 2 * self.l[j])
            )
            w[..., j] = C * self.l[j] ** x[..., j] * (1 - self.l[j]) ** (1 - x[..., j])
        return np.where(in_support, np.prod(w, -1), 0.0)

    def _spawn(self, sampler, dimension):
        if dimension == self.d:  # don't do anything if the dimension doesn't change
            spawn = BernoulliCont(sampler, lam=self.lam)
        else:
            l0 = self.l[0]
            if (self.l != l0).any():
                raise DimensionError(
                    """
                        In order to spawn a BernoulliCont measure
                        lam must all be the same."""
                )
            spawn = BernoulliCont(sampler, lam=l0)
        return spawn
