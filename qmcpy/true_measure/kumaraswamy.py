from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from scipy.special import betaln
from scipy.sparse import diags
import numpy as np


class Kumaraswamy(AbstractTrueMeasure):
    r"""
    Kumaraswamy distribution as described in [https://en.wikipedia.org/wiki/Kumaraswamy_distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution).

    Examples:
        >>> true_measure = Kumaraswamy(DigitalNetB2(2,seed=7),a=[1,2],b=[3,4])
        >>> true_measure(4)
        array([[0.34705366, 0.6782161 ],
               [0.0577568 , 0.36189538],
               [0.76344358, 0.0932949 ],
               [0.17065545, 0.43009386]])

        The covariance is diagonal, so it is stored and shown in sparse form.

        >>> true_measure  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Kumaraswamy (AbstractTrueMeasure)
            a               [1 2]
            b               [3 4]
            mean            [0.25  0.406]
            variance        [0.037 0.035]
            standard_deviation [0.194 0.187]
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                with 2 stored elements (1 diagonals) and shape (2, 2)>
                Coords Values
                (0, 0) 0.0374...
                (1, 1) 0.0348...

        With independent replications

        >>> x = Kumaraswamy(DigitalNetB2(3,seed=7,replications=2),a=[1,2,3],b=[3,4,5])(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.09004177, 0.22144305, 0.62190133],
                [0.31710078, 0.48718217, 0.47325643],
                [0.19657641, 0.57423463, 0.25697057],
                [0.56103074, 0.28939035, 0.63654112]],
        <BLANKLINE>
               [[0.18006788, 0.62226635, 0.5083556 ],
                [0.22602452, 0.10519477, 0.42823814],
                [0.08428482, 0.28804621, 0.2414302 ],
                [0.37253319, 0.45379743, 0.63366422]]])
    """

    def __init__(self, sampler, a=2, b=2):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            a (Union[float, np.ndarray]): First parameter $\alpha > 0$.
            b (Union[float, np.ndarray]): Second parameter $\beta > 0$.
        """
        self.parameters = ["a", "b", "mean", "variance", "standard_deviation", "covariance"]
        self.domain = np.array([[0, 1]])
        self.range = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self.a = a
        self.b = b
        self.alpha = np.array(a)
        if self.alpha.size == 1:
            self.alpha = self.alpha.item() * np.ones(self.d)
        self.beta = np.array(b)
        if self.beta.size == 1:
            self.beta = self.beta.item() * np.ones(self.d)
        if not (self.alpha.shape == (self.d,) and self.beta.shape == (self.d,)):
            raise DimensionError(
                "a and b must be scalar or have length equal to dimension."
            )
        if not (
            np.isfinite(self.alpha).all()
            and np.isfinite(self.beta).all()
            and (self.alpha > 0).all()
            and (self.beta > 0).all()
        ):
            raise ParameterError("Kumaraswamy requires finite a,b>0.")

        mean, variance = self._compute_moments()
        self._set_moments(
            mean=mean,
            variance=variance,
            standard_deviation=np.sqrt(variance),
            covariance=diags(variance, format="dia"),
        )
        super(Kumaraswamy, self).__init__()
        assert self.alpha.shape == (self.d,) and self.beta.shape == (self.d,)

    def _compute_moments(self):
        r"""
        Compute the marginal mean and variance of each coordinate.

        The Kumaraswamy raw moments are $M_n = b\,B(1 + n/a, b)$ [1], so the
        mean is $M_1$ and the variance is $M_2 - M_1^2$. Forming that difference
        directly causes cancellation error once the variance is small relative
        to $M_1^2$ (e.g. large $a$).

        Instead, with the log-moment function $K(r) = \log M_r$,

        $$\text{mean} = e^{K(1)}, \qquad
          \operatorname{Var}[X] = \text{mean}^2\,(e^{q} - 1), \qquad
          q = K(2) - 2K(1).$$

        Each log-moment is available in closed form via the log-Beta function
        [2], $K(r) = \log b + \ln B(1 + r/a, b)$, so ``mean`` and $q$ are
        evaluated exactly (up to floating-point rounding of ``betaln``) for
        every $a, b > 0$. The $\log b$ terms cancel in $q = \ln B(1 + 2/a, b) -
        2\ln B(1 + 1/a, b) - \log b$. Because $K$ is convex ($r \mapsto M_r$ is
        log-convex by Holder's inequality [3]) we have $q \ge 0$, so ``expm1``
        [4] recovers $e^{q} - 1$ without cancellation even when $q$ is tiny.
        Every operation is elementwise on the per-coordinate parameters $a$ and
        $b$, so ``mean`` and ``variance`` are returned as length-``d`` arrays.

        **References:**

        1.  Kumaraswamy distribution. Wikipedia.
            [https://en.wikipedia.org/wiki/Kumaraswamy_distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution).

        2.  SciPy Reference. scipy.special.betaln.
            [https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betaln.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.betaln.html).

        3.  G. H. Hardy, J. E. Littlewood, and G. Polya.
            Inequalities, 2nd edition, Cambridge University Press, Cambridge, 1952
            (Holder's inequality; implies log-convexity of the moment sequence).

        4.  NumPy Reference. numpy.expm1.
            [https://numpy.org/doc/stable/reference/generated/numpy.expm1.html](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html).

        Returns:
            tuple: Length ``d`` arrays ``(mean, variance)``.
        """
        inv_a = 1.0 / self.alpha
        beta = self.beta

        # K(r) = log M_r = log(b) + betaln(1 + r/a, b), the log of the r-th raw moment.
        log_b = np.log(beta)
        k1 = log_b + betaln(1.0 + inv_a, beta)
        k2 = log_b + betaln(1.0 + 2.0 * inv_a, beta)

        mean = np.exp(k1)

        # q = K(2) - 2K(1) >= 0 by log-convexity of the moment sequence.
        # expm1(q) accurately computes exp(q) - 1 when q is very small.
        q = k2 - 2.0 * k1
        variance = mean * mean * np.expm1(q)

        return mean, variance

    def _transform(self, x):
        return (1 - (1 - x) ** (1 / self.beta)) ** (1 / self.alpha)

    def _weight(self, x):
        return np.prod(
            self.alpha
            * self.beta
            * x ** (self.alpha - 1)
            * (1 - x**self.alpha) ** (self.beta - 1),
            -1,
        )

    def _spawn(self, sampler, dimension):
        if dimension == self.d:  # don't do anything if the dimension doesn't change
            spawn = Kumaraswamy(sampler, a=self.alpha, b=self.beta)
        else:
            a = self.alpha[0]
            b = self.beta[0]
            if not (all(self.alpha == a) and all(self.beta == b)):
                raise DimensionError(
                    """
                    In order to spawn a Kumaraswamy measure
                    a must all be the same and
                    b must all be the same"""
                )
            spawn = Kumaraswamy(sampler, a=a, b=b)
        return spawn
