from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from scipy.sparse import diags
import numpy as np


class Uniform(AbstractTrueMeasure):
    r"""
    Uniform distribution, see [https://en.wikipedia.org/wiki/Continuous_uniform_distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution).

    Examples:
        >>> true_measure = Uniform(DigitalNetB2(2,seed=7),lower_bound=[0,.5],upper_bound=[2,3])
        >>> true_measure(4)
        array([[1.44324713, 2.7873875 ],
               [0.32691107, 1.5741214 ],
               [1.97352511, 0.58590959],
               [0.8591331 , 1.89690854]])

        The covariance is diagonal, so it is stored and shown in sparse form.

        >>> true_measure  # doctest: +NORMALIZE_WHITESPACE
        Uniform (AbstractTrueMeasure)
            lower_bound     [0.  0.5]
            upper_bound     [2 3]
            mean            [1.   1.75]
            variance        [0.333 0.521]
            standard_deviation [0.577 0.722]
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                with 2 stored elements (1 diagonals) and shape (2, 2)>
                Coords Values
                (0, 0) 0.3333333333333333
                (1, 1) 0.5208333333333334

        With independent replications

        >>> x = Uniform(DigitalNetB2(3,seed=7,replications=2),lower_bound=[.25,.5,.75],upper_bound=[1.75,1.5,1.25])(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.61979915, 0.6821862 , 1.12366296],
                [1.27229355, 1.16169442, 0.9644598 ],
                [0.97209782, 1.29818233, 0.79100643],
                [1.62311988, 0.79520621, 1.13747905]],
        <BLANKLINE>
               [[0.92315337, 1.35899604, 1.0027484 ],
                [1.05453886, 0.54353443, 0.91782473],
                [0.59821215, 0.79281506, 0.78420518],
                [1.37943573, 1.10241448, 1.13481488]]])
    """

    def __init__(self, sampler, lower_bound=0, upper_bound=1):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            lower_bound (Union[float, np.ndarray]): Lower bound.
            upper_bound (Union[float, np.ndarray]): Upper bound.
        """
        self.parameters = ["lower_bound", "upper_bound", "mean", "variance", "standard_deviation", "covariance"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if np.isscalar(self.lower_bound):
            lower_bound = np.tile(self.lower_bound, self.d)
        if np.isscalar(self.upper_bound):
            upper_bound = np.tile(self.upper_bound, self.d)
        self.a = np.array(lower_bound, dtype=np.float64)
        self.b = np.array(upper_bound, dtype=np.float64)
        if len(self.a) != self.d or len(self.b) != self.d:
            raise DimensionError(
                "upper bound and lower bound must be of length dimension"
            )
        if not (np.all(np.isfinite(self.a)) and np.all(np.isfinite(self.b))):
            raise ParameterError("upper bound and lower bound must be finite")
        self.delta = self.b - self.a
        if np.any(self.delta <= 0):
            raise ParameterError(
                "upper bound must be strictly greater than lower bound"
            )
        mean = (self.a + self.b) / 2
        variance = self.delta**2 / 12
        self._set_moments(
            mean=mean,
            variance=variance,
            standard_deviation=self.delta / np.sqrt(12),
            covariance=diags(variance, format="dia"),
        )
        self.inv_delta_prod = 1 / self.delta.prod()
        self.range = np.hstack(
            (self.a.reshape((self.d, 1)), self.b.reshape((self.d, 1)))
        )
        super(Uniform, self).__init__()
        assert self.a.shape == (self.d,) and self.b.shape == (self.d,)

    def _transform(self, x):
        return x * self.delta + self.a

    def _weight(self, x):
        return np.tile(self.inv_delta_prod, x.shape[:-1])

    def _spawn(self, sampler, dimension):
        if dimension == self.d:  # don't do anything if the dimension doesn't change
            spawn = Uniform(sampler, lower_bound=self.a, upper_bound=self.b)
        else:
            l = self.a[0]
            u = self.b[0]
            if not (all(self.a == l) and all(self.b == u)):
                raise DimensionError(
                    """
                    In order to spawn a uniform measure with different dimensions
                    the lower bounds must all be the same and 
                    the upper bounds must all be the same"""
                )
            spawn = Uniform(sampler, lower_bound=l, upper_bound=u)
        return spawn
