from .abstract_discrete_distribution import AbstractDiscreteDistribution
import numpy as np
from qmcpy.util import ParameterError, ParameterWarning
import warnings


class LatinHypercube(AbstractDiscreteDistribution):
    r"""
    Latin Hypercube Sampler for quasi-Monte Carlo and experimental design.

    Latin Hypercube Sampling (LHS) generates points with excellent univariate
    stratification: splitting $[0,1)$ into `n` equal strata along *any* single
    coordinate axis places exactly one point in each stratum. Introduced by
    McKay, Beckman, and Conover as a variance-reduction alternative to simple
    random sampling for computer experiments, LHS is asymptotically at least
    as accurate as Monte Carlo for the additive part of an integrand, with the
    rate of improvement characterized by Stein and later by Loh via a
    multivariate central limit theorem.

    Note:
        - Unlike the low discrepancy sequences in this package (e.g. `Lattice`,
          `Halton`, `DigitalNetB2`), `LatinHypercube` points are *not* extensible
          in `n`: the entire point set must be regenerated whenever `n` changes,
          since the strata boundaries themselves depend on `n`. 
          Consequently `LatinHypercube` requires `n_min=0`, it cannot be generated starting from a nonzero offset.
        - `replications` produces independent randomizations (independent random
          permutations, and independent within-stratum jitter when `randomize`
          is `True`), not an extension or reshaping of a single sequence.
        - When `randomize` is `False`, points sit at the *center* of their
          stratum instead of a uniformly jittered position within it. The
          assignment of strata to dimensions (the permutation) is still drawn
          randomly in this case -- without it, every dimension would place its
          points on the same diagonal pattern, which is not a useful point set.
          Only the *within-stratum* position becomes deterministic.

    Examples:
        >>> discrete_distrib = LatinHypercube(2,replications=None,seed=7)
        >>> discrete_distrib(4)
        array([[0.10079093, 0.46583043],
               [0.73559373, 0.81194835],
               [0.44928008, 0.53874559],
               [0.9427258 , 0.10932748]])

        Replications of independent randomizations

        >>> x = LatinHypercube(3,replications=2,seed=7)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.10407119, 0.26180415, 0.40801808],
                [0.599802  , 0.86538519, 0.71594779],
                [0.41110958, 0.60764889, 0.84352474],
                [0.91828595, 0.20655501, 0.12138689]],
        <BLANKLINE>
               [[0.12951706, 0.17802722, 0.63472096],
                [0.9652066 , 0.31881758, 0.78031776],
                [0.3523449 , 0.55568244, 0.26354665],
                [0.74188086, 0.98836479, 0.12774161]]])

        Centered (non-randomized) points: each point sits at the middle of
        its stratum instead of a random position within it

        >>> LatinHypercube(2,replications=None,seed=7,randomize=False)(4)
        array([[0.125, 0.375],
               [0.625, 0.875],
               [0.375, 0.625],
               [0.875, 0.125]])


    **References:**

    1.  M. D. McKay, R. J. Beckman, and W. J. Conover.  
        A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code.  
        Technometrics, 21(2):239-245, 1979.  
        [https://doi.org/10.1080/00401706.1979.10489755](https://doi.org/10.1080/00401706.1979.10489755).

    2.  M. Stein.  
        Large Sample Properties of Simulations Using Latin Hypercube Sampling.  
        Technometrics, 29(2):143-151, 1987.  
        [https://doi.org/10.1080/00401706.1987.10488205](https://doi.org/10.1080/00401706.1987.10488205).

    3.  A. B. Owen.  
        Controlling Correlations in Latin Hypercube Samples.  
        Journal of the American Statistical Association, 89(428):1517-1522, 1994.  
        [https://doi.org/10.1080/01621459.1994.10476891](https://doi.org/10.1080/01621459.1994.10476891).

    4.  W.-L. Loh.  
        On Latin Hypercube Sampling.  
        The Annals of Statistics, 24(5):2058-2080, 1996.  
        [https://doi.org/10.1214/aos/1069362310](https://doi.org/10.1214/aos/1069362310).

    5.  B. Tang.  
        Orthogonal Array-Based Latin Hypercubes.  
        Journal of the American Statistical Association, 88(424):1392-1397, 1993.  
        [https://doi.org/10.1080/01621459.1993.10476423](https://doi.org/10.1080/01621459.1993.10476423).
    """

    def __init__(
            self, dimension, replications, seed, randomize="TRUE"
            ):
        r"""
        Args:
            dimension (int): Dimension of the samples.

            replications (Union[None, int]): Number of independent LHS designs
                to generate. Each replication is its own independently permuted,
                independently jittered stratification into `n` strata.

            seed (Union[None, int, np.random.SeedSequence]): Seed for the random
                number generator to ensure reproducibility.

            randomize (str): Whether to jitter each point uniformly within its
                stratum (`True`, the default) or place it at the stratum's
                center (`False`), must be one of 'TRUE', 'FALSE', 'NONE', or 'NO' (case-insensitive).
    """
        super().__init__(dimension=dimension, replications=replications, seed=seed, d_limit=np.inf, n_limit=np.inf)
        self.randomize = str(randomize).upper()
        if self.randomize in ("NONE", "NO", "FALSE"):
            self.randomize = "FALSE"
        elif self.randomize == "TRUE":
            self.randomize = "TRUE"
        else:
            raise ParameterError(
                f"randomize must be one of 'TRUE', 'FALSE', 'NONE', or 'NO' (case-insensitive), got {randomize!r}."
            )

    def _gen_samples(
            self, n=None, n_min=None, n_max=None, return_binary=False, warn=True
        ):
        r"""..."""  # (inchangee)
        if return_binary:
            raise ParameterError("LatinHypercube does not support return_binary=True")
        if n_min != 0:
            raise ParameterError(
            "LatinHypercube requires n_min=0: since the strata boundaries "
            "depend on the total number of points n, points cannot be "
            "generated starting from a nonzero index."
        )
        if warn and self.randomize == "FALSE":
            warnings.warn(
            "randomize=False only fixes the position of each point within "
            "its stratum (center instead of jittered). The assignment of "
            "strata to dimensions is still drawn randomly and depends on "
            "seed.",
            ParameterWarning,
        )
        n = int(n_max - n_min)
        keys = self.rng.random(size=(self.replications, self.d, n))
        perm_indices = np.argsort(keys, axis=-1)
        permutations = perm_indices + 1
        if self.randomize == "TRUE":
            U = self.rng.uniform(0, 1, size=permutations.shape)
            result = (permutations - U) / n
        else:
            result = (permutations - 0.5) / n
        return result.transpose(0, 2, 1)

    def _spawn(self, child_seed, dimension):
        return LatinHypercube(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            randomize=self.randomize,
        )

    def __repr__(self):
        return super().__repr__("LatinHypercube")