import numpy as np
from qmcpy.util import ParameterError, ParameterWarning
from pathlib import Path
import qmctoolscl
import warnings
from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from functools import lru_cache

@lru_cache(maxsize=1)
def load_korobov_table(
        npz_path=Path(__file__).resolve().parent / "generating_params" / "korobov_p2_table.npz"
    ):
    """Load the Korobov table from the compressed .npz file. Cached via
       lru_cache: the file is only actually read once per process, with no
       explicit module-level global variable."""
    with np.load(npz_path) as data:
        raw = data["raw"]
        lut = {
            "n_values": data["n_values"],
            "d_values": data["d_values"],
            "a": data["a"],
            "p2": data["p2"],
            "exact": data["exact"],
        }
    return raw, lut

def get_a(lut, n, d):
    i = np.searchsorted(lut["n_values"], n)
    if i >= len(lut["n_values"]) or lut["n_values"][i] != n:
        raise ParameterError(
            f"KorobovLattice: n={n} is not tabulated. Available n: "
            f"{lut['n_values'].tolist()}"
        )
    j = np.searchsorted(lut["d_values"], d)
    if j >= len(lut["d_values"]) or lut["d_values"][j] != d:
        raise ParameterError(
            f"KorobovLattice: d={d} is not tabulated (table covers d = "
            f"{lut['d_values'][0]}..{lut['d_values'][-1]})"
        )
    return int(lut["a"][i, j])



class KorobovLattice(AbstractLDDiscreteDistribution):
    r"""
    Korobov lattice rule with a tabulated, quality-optimized generating parameter.

    A rank-1 lattice rule with $n$ points and generating vector $z\in\mathbb{Z}^d$ is
    $P_n(z) = \{(\{k z_1/n\},\dots,\{k z_d/n\}) : k=0,\dots,n-1\}$. The Korobov
    construction restricts $z$ to a single integer parameter $a$:
    $z(a) = (1,a,a^2,\dots,a^{d-1}) \bmod n$, with $\gcd(a,n)=1$.

    Rather than searching for $a$ at construction time, this class looks up $a$
    in a precomputed table, for every $(n,d)$ pair in the table, minimizing the
    weighted $P_2$ figure of merit (the squared worst-case integration error in
    the weighted Korobov space of smoothness 2) with product weights
    $\gamma_j = 1/j^2$.

    Note:
        - Because the optimal $a$ depends on the *total* number of points $n$,
          a Korobov lattice cannot be incrementally extended the way `Lattice`
          can: `n_min` must be 0, and `n` must be one of the values in the
          precomputed table (a `ParameterError` is raised otherwise, listing
          the available values).
        - The table covers $d = 1,\dots,250$ and $n$ up to $131072$, on a grid
          of powers of two, the largest prime below each power of two, and a
          set of round primes.
        - The first point of an unrandomized Korobov lattice is the origin.
        - `replications` only randomizes independent Cranley-Patterson shifts
          of the *same* underlying deterministic lattice; it does not draw
          independent generating vectors.

    Examples:
        >>> discrete_distrib = KorobovLattice(2,seed=7)
        >>> discrete_distrib(8)
        array([[0.04386058, 0.58727432],
               [0.16886058, 0.96227432],
               [0.29386058, 0.33727432],
               [0.41886058, 0.71227432],
               [0.54386058, 0.08727432],
               [0.66886058, 0.46227432],
               [0.79386058, 0.83727432],
               [0.91886058, 0.21227432]])

        Replications of independent randomizations

        >>> x = KorobovLattice(3,seed=7,replications=2)(8)
        >>> x.shape
        (2, 8, 3)
        >>> x
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.16886058, 0.96227432, 0.4941824 ],
                [0.29386058, 0.33727432, 0.6191824 ],
                [0.41886058, 0.71227432, 0.7441824 ],
                [0.54386058, 0.08727432, 0.8691824 ],
                [0.66886058, 0.46227432, 0.9941824 ],
                [0.79386058, 0.83727432, 0.1191824 ],
                [0.91886058, 0.21227432, 0.2441824 ]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.77712985, 0.07169968, 0.23105352],
                [0.90212985, 0.44669968, 0.35605352],
                [0.02712985, 0.82169968, 0.48105352],
                [0.15212985, 0.19669968, 0.60605352],
                [0.27712985, 0.57169968, 0.73105352],
                [0.40212985, 0.94669968, 0.85605352],
                [0.52712985, 0.32169968, 0.98105352]]])

        Unrandomized Korobov lattice

        >>> KorobovLattice(2,randomize="FALSE",seed=7)(8,warn=False)
        array([[0.   , 0.   ],
               [0.125, 0.375],
               [0.25 , 0.75 ],
               [0.375, 0.125],
               [0.5  , 0.5  ],
               [0.625, 0.875],
               [0.75 , 0.25 ],
               [0.875, 0.625]])

    **References:**

    1.  N. M. Korobov.
        The approximate computation of multiple integrals.
        Dokl. Akad. Nauk SSSR, 124:1207-1210. 1959.

    2.  I. H. Sloan and S. Joe.
        Lattice Methods for Multiple Integration.
        Oxford University Press. 1994.

    3.  J. Dick, F. Y. Kuo, and I. H. Sloan.
        High-dimensional integration: the quasi-Monte Carlo way.
        Acta Numerica, 22:133-288. 2013.
        [https://doi.org/10.1017/S0962492913000044](https://doi.org/10.1017/S0962492913000044).
    """
    def __init__(
            self,
            dimension=1,
            replications=None,
            seed=None,
            randomize="SHIFT",
        ):
        r"""
        Args:
            dimension (int): Dimension of the samples. Must be between 1 and
                250 (the range covered by the precomputed table).

            replications (int): Number of independent Cranley-Patterson
                shifts of the same underlying deterministic lattice.

            seed (Union[None, int, np.random.SeedSequence]): Seed the random
                number generator for reproducibility.

            randomize (str): Options are

                - `'SHIFT'` or `'TRUE'`: Random Cranley-Patterson shift (the default).
                - `'FALSE'`, `'NONE'`, or `'NO'`: No randomization. In this
                case the first point will be the origin.
    """
        super().__init__(dimension, replications, seed, d_limit = 250, n_limit = 131072)

        self.randomize = str(randomize).upper()
        if self.randomize == "TRUE":
            self.randomize = "SHIFT"
        if self.randomize == "NONE":
            self.randomize = "FALSE"
        if self.randomize == "NO":
            self.randomize = "FALSE"
        assert self.randomize in ["SHIFT", "FALSE"]
        if self.randomize not in ("SHIFT", "FALSE"):
            raise ParameterError(
            f"randomize must be one of 'SHIFT', 'TRUE', 'FALSE', 'NONE', or 'NO' (case-insensitive), got {randomize!r}."
            )
        if self.randomize == "SHIFT":
            self.shift = self.rng.uniform(size=(self.replications, self.d))

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if return_binary:
            raise ParameterError("KorobovLattice does not support return_binary=True")

        if n_min != 0:
            raise ParameterError(
                "KorobovLattice requires n_min=0: the optimal parameter a "
                "depends on the total number of points n, so a Korobov lattice "
                "cannot be incrementally extended like Lattice can."
            )

        if n_min == 0 and self.randomize == "FALSE" and warn:
            warnings.warn(
                "Without randomization, the first lattice point is the origin",
                ParameterWarning,
            )
        # Loading the table
        _RAW, _LUT = load_korobov_table()

        n = int(n_max - n_min)
        d = int(self.d)
        a = get_a(_LUT, n, d)

        z = np.empty(d, dtype=np.int64)
        p = 1
        for j in range(d):
            z[j] = p
            p = (p * a) % n

        k = np.arange(n, dtype=np.int64)[:, None]           # (n, 1)
        x = ((k * z[None, :]) % n) / n                       # (n, d)
        x = x[None, :, :].astype(np.float64)                 # (1, n, d) -- r_x = 1

        r_x = np.uint64(1)
        n_u = np.uint64(n)
        d_u = np.uint64(d)

        if self.randomize == "FALSE":
            xr = x
        elif self.randomize == "SHIFT":
            r = np.uint64(self.replications)
            xr = np.empty((r, n, d), dtype=np.float64)
            qmctoolscl.lat_shift_mod_1(r, n_u, d_u, r_x, x, self.shift, xr, backend="c")
        return xr

    def _spawn(self, child_seed, dimension):
        return KorobovLattice(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            randomize=self.randomize,
        )