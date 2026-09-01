import numpy as np
from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from ..util import ParameterError


class Ishigami(AbstractIntegrand):
    r"""
    Ishigami function in $d=3$ dimensions from [1] and [https://www.sfu.ca/~ssurjano/ishigami.html](https://www.sfu.ca/~ssurjano/ishigami.html).

    $$g(\boldsymbol{t}) = (1+bt_2^4)\sin(t_0)+a\sin^2(t_1), \qquad \boldsymbol{T} = (T_0,T_1,T_2) \sim \mathcal{U}(-\pi,\pi)^3.$$

    Examples:
        >>> integrand = Ishigami(DigitalNetB2(3,seed=7))
        >>> y = integrand(2**10)
        >>> y.shape
        (1024,)
        >>> print("%.4f"%y.mean())
        3.5000
        >>> integrand.true_measure  # doctest: +NORMALIZE_WHITESPACE
        Uniform (AbstractTrueMeasure)
            lower_bound     -3.142
            upper_bound     3.142
            mean            [0. 0. 0.]
            variance        [3.29 3.29 3.29]
            standard_deviation [1.814 1.814 1.814]
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                             with 3 stored elements (1 diagonals) and shape (3, 3)>
                              Coords Values
                              (0, 0) 3.289868133696453
                              (1, 1) 3.289868133696453
                              (2, 2) 3.289868133696453

        With independent replications

        >>> integrand = Ishigami(DigitalNetB2(3,seed=7,replications=2**4))
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1)
        >>> muhats.shape
        (16,)
        >>> print("%.4f"%muhats.mean())
        3.4646

    **References:**

    1.  Ishigami, T., & Homma, T.
        An importance quantification technique in uncertainty analysis for computer models.
        In Uncertainty Modeling and Analysis, 1990.
        Proceedings, First International Symposium on (pp. 398-403). IEEE.
    """

    def __init__(self, sampler, a=7, b=0.1):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            a (float): First parameter $a$.
            b (float): Second parameter $b$.
        """
        self.sampler = sampler
        if self.sampler.d != 3:
            raise ParameterError("Ishigami integrand requires 3 dimensional sampler")
        self.a = a
        self.b = b
        self.true_measure = Uniform(self.sampler, lower_bound=-np.pi, upper_bound=np.pi)
        super(Ishigami, self).__init__(
            dimension_indv=(), dimension_comb=(), parallel=False
        )

    def g(self, t):
        y = (1 + self.b * t[..., 2] ** 4) * np.sin(t[..., 0]) + self.a * np.sin(
            t[..., 1]
        ) ** 2
        return y

    def _spawn(self, level, sampler):
        return Ishigami(sampler=sampler, a=self.a, b=self.b)

    @staticmethod
    def _exact_sensitivity_indices(indices, a, b):
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        assert a.shape == b.shape and a.ndim == 1 and b.ndim == 1
        mu = a / 2
        m2 = 1 / 2 + 3 / 8 * a**2 + np.pi**4 / 5 * b + np.pi**8 / 18 * b**2
        tau_closed = {
            repr([True, False, False]): 1 / 50 * (5 + np.pi**4 * b) ** 2,
            repr([False, True, False]): a**2 / 8,
            repr([False, False, True]): 0,
            repr([True, True, False]): a**2 / 8 + 1 / 50 * (5 + np.pi**4 * b) ** 2,
            repr([True, False, True]): 1
            / 90
            * (45 + 18 * np.pi**4 * b + 5 * np.pi**8 * b**2),
            repr([False, True, True]): a**2 / 8,
        }
        tau_total = {
            repr([False, True, True]): a**2 / 8 + 8 * np.pi**8 / 225 * b**2,
            repr([True, False, True]): 1
            / 90
            * (45 + 18 * np.pi**4 * b + 5 * np.pi**8 * b**2),
            repr([True, True, False]): 1 / 2
            + a**2 / 8
            + np.pi**4 / 5 * b
            + np.pi**8 / 18 * b**2,
            repr([False, False, True]): 8 * np.pi**8 / 225 * b**2,
            repr([False, True, False]): a**2 / 8,
            repr([True, False, False]): 1
            / 90
            * (45 + 18 * np.pi**4 * b + 5 * np.pi**8 * b**2),
        }
        solution = np.zeros((2, len(indices), len(a)), dtype=float)
        for j, idx in enumerate(indices):
            solution[0, j] = tau_closed[repr(idx.tolist())] / (m2 - mu**2)
            solution[1, j] = tau_total[repr(idx.tolist())] / (m2 - mu**2)
        return solution.squeeze()

    @staticmethod
    def _exact_fu_functions(x, indices, a, b):
        x = np.atleast_2d(x)
        n = len(x)
        a, b = np.atleast_1d(a), np.atleast_1d(b)
        assert x.ndim == 2 and x.shape == (n, 3) and a.shape == (1,) and b.shape == (1,)
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
        fus = {
            repr([]): a / 2,
            repr([0]): -1 / 5 * (5 + np.pi**4 * b) * np.sin(2 * np.pi * x0),
            repr([1]): -1 / 2 * a * np.cos(4 * np.pi * x1),
            repr([2]): 0,
            repr([0, 1]): 0,
            repr([0, 2]): -4
            / 5
            * b
            * np.pi**4
            * (1 + 10 * (-1 + x2) * x2 * (1 + 2 * (-1 + x2) * x2))
            * np.sin(2 * np.pi * x0),
            repr([1, 2]): -1 / 2 * a * np.cos(4 * np.pi * x1)
            + 1 / 5 * (5 + b * np.pi**4) * np.sin(2 * np.pi * x2),
            repr([0, 1, 2]): 1 / 2 * a * np.cos(4 * np.pi * x1)
            - 1 / 5 * (5 + b * np.pi**4) * np.sin(2 * np.pi * x1),
        }
        solution = np.zeros((n, len(indices)), dtype=float)
        for j, idx in enumerate(indices):
            solution[:, j] = fus[repr(idx)]
        return solution.squeeze()
