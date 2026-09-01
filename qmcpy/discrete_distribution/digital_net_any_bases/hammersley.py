from qmcpy.util import ParameterError,ParameterWarning
import numpy as np
from .halton import Halton
from qmcpy.discrete_distribution.abstract_discrete_distribution import AbstractLDDiscreteDistribution
from .digital_net_any_bases import DigitalNetAnyBases
import warnings



class Hammersley(DigitalNetAnyBases):
    r"""
    Hammersley point set: a deterministic, 'closed' low discrepancy point set.

    With $p_1,\dots,p_{d-1}$ the first $d-1$ prime numbers, the point set
    $\{t_0,\dots,t_{n-1}\}$ with $n$ points in $d$ dimensions is given by
    $t_i = (i/n,\ \varphi_{p_1}(i),\ \dots,\ \varphi_{p_{d-1}}(i))$
    for $i=0,\dots,n-1$, where $\varphi_p$ denotes the radical inverse
    function in base $p$.

    Being a 'closed' point set (n must be fixed in advance, unlike an
    extensible sequence such as Halton), the QMC error bound gains one
    fewer power of $\log n$ than the corresponding Halton bound:
    $|I_d(f)-Q_{n,d}(f)| \le C_d\, (\log n)^{d-1}/n\, V(f)$.

    Note:
        - This class is fully deterministic: no randomization is supported,
          and the `seed` argument has no effect on the generated points.
        - The first point is always the origin.
        - Because the $i/n$ coordinate depends on the *total* number of
          points $n$, this point set cannot be incrementally extended the
          way `Halton` can: `n_min` must be 0.
        - `dimension` must be an `int`: unlike `Halton`, the $i/n$
          coordinate is not associated with any prime index, so "component
          at index j" would be ambiguous for an array-valued `dimension`.

    Examples:
        >>> discrete_distrib = Hammersley(4,seed=7)
        >>> discrete_distrib(8,warn=False)
        array([[0.        , 0.        , 0.        , 0.        ],
               [0.125     , 0.5       , 0.33333333, 0.2       ],
               [0.25      , 0.25      , 0.66666667, 0.4       ],
               [0.375     , 0.75      , 0.11111111, 0.6       ],
               [0.5       , 0.125     , 0.44444444, 0.8       ],
               [0.625     , 0.625     , 0.77777778, 0.04      ],
               [0.75      , 0.375     , 0.22222222, 0.24      ],
               [0.875     , 0.875     , 0.55555556, 0.44      ]])

        dimension=1 : only the i/n coordinate

        >>> Hammersley(1)(4,warn=False)
        array([[0.  ],
               [0.25],
               [0.5 ],
               [0.75]])

    **References:**

    1.  J. Dick, F. Y. Kuo, and I. H. Sloan.
        High-dimensional integration: the quasi-Monte Carlo way.
        Acta Numerica, 22:133-288. 2013.
        [https://doi.org/10.1017/S0962492913000044](https://doi.org/10.1017/S0962492913000044).

    2.  J. M. Hammersley.
        Monte Carlo methods for solving multivariate problems.
        Annals of the New York Academy of Sciences, 86(3):844-874. 1960.
    """

    def __init__(self,
                 dimension=1,
                 seed=None,
                 t=None,
                 n_lim=2**32,
                 warn = True
                ):
        r"""
        Args:
            dimension (int): Dimension of the samples. Must be a scalar
                `int` (unlike `Halton`, an array of indices is not
                supported -- see class Notes).

            seed (Union[None, int, np.random.SeedSequence]): Unused; kept
                for API consistency with the other discrete distributions.
                This point set is fully deterministic, so `seed` has no
                effect on the generated points.

            t (Union[None, int]): Passed through to the internal `Halton`
                generator used for dimensions 2,...,`dimension` (ignored
                when `dimension` is 1). See `Halton`'s docstring for
                details.

            n_lim (int): Maximum number of points `n` this distribution
                can be asked to generate.
        """

        if not np.isscalar(dimension):
            raise ParameterError(
                "Hammersley does not support dimension as an array of "
                "indices: unlike Halton, the i/n coordinate is not associated "
                "with any prime index, so 'component at index j' is ambiguous "
                "for this construction. Pass an int instead."
            )
        dimension = int(dimension)
        if dimension < 1:
            raise ParameterError("Hammersley requires dimension >= 1")

        AbstractLDDiscreteDistribution.__init__(
            self, dimension, replications=None, seed=seed,
            d_limit=10**9, n_limit=n_lim,
        )

        if dimension > 1:
            self.halton = Halton(
                dimension - 1,
                replications=None,
                seed=seed,
                randomize='None',
                t=t,
                n_lim=n_lim,
                warn=False)
        else:
            self.halton = None
        self.warn = warn

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if return_binary:
            raise ParameterError("Hammersley does not support return_binary=True")
        if n_min != 0:
            raise ParameterError(
                "Hammersley requires n_min=0: the i/n coordinate "
                "depends on the total number of points n."
            )
        if warn:
            warnings.warn(
                "Hammersley is deterministic; the first point is "
                "always the origin",
                ParameterWarning,
            )

        n = int(n_max - n_min)
        grid = (1 / n) * np.arange(n, dtype=np.float64)
        grid = grid[None, :, None]                     # (1, n, 1)

        if self.halton is None:
            x = grid
        else:
            rest = self.halton.gen_samples(n_min=n_min, n_max=n_max,
                                            return_binary=False, warn=False)
            rest = rest[None, :, :]                      # (1, n, d-1)
            x = np.concatenate([grid, rest], axis=-1)    # (1, n, d)
        return x

    def _spawn(self, child_seed, dimension):
        return Hammersley(
            dimension=dimension,
            seed=child_seed,
        )
