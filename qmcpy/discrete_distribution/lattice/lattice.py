from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError, ParameterWarning
import qmctoolscl
import numpy as np
from numpy.lib.npyio import DataSource
from os.path import dirname, abspath
import warnings
from copy import deepcopy


class Lattice(AbstractLDDiscreteDistribution):
    r"""
    Low discrepancy lattice sequence.

    Note:
        - Lattice sample sizes should be powers of $2$ e.g. $1$, $2$, $4$, $8$, $16$, $\dots$.
        - The first point of an unrandomized lattice is the origin.

    Examples:
        >>> discrete_distrib = Lattice(2,seed=7)
        >>> discrete_distrib(4)
        array([[0.04386058, 0.58727432],
               [0.54386058, 0.08727432],
               [0.29386058, 0.33727432],
               [0.79386058, 0.83727432]])
        >>> discrete_distrib(1) # first point in the sequence
        array([[0.04386058, 0.58727432]])
        >>> discrete_distrib
        Lattice (AbstractLDDiscreteDistribution)
            d               2^(1)
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           RADICAL INVERSE
            n_limit         2^(20)
            entropy         7

        Replications of independent randomizations

        >>> x = Lattice(3,seed=7,replications=2)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.54386058, 0.08727432, 0.8691824 ],
                [0.29386058, 0.33727432, 0.1191824 ],
                [0.79386058, 0.83727432, 0.6191824 ]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.15212985, 0.19669968, 0.60605352],
                [0.90212985, 0.44669968, 0.85605352],
                [0.40212985, 0.94669968, 0.35605352]]])


        Different orderings (avoid warnings that the first point is the origin).

        >>> Lattice(dimension=2,randomize=False,order='RADICAL INVERSE')(4,warn=False)
        array([[0.  , 0.  ],
               [0.5 , 0.5 ],
               [0.25, 0.75],
               [0.75, 0.25]])
        >>> Lattice(dimension=2,randomize=False,order='GRAY')(4,warn=False)
        array([[0.  , 0.  ],
               [0.5 , 0.5 ],
               [0.75, 0.25],
               [0.25, 0.75]])
        >>> Lattice(dimension=2,randomize=False,order='LINEAR')(4,warn=False)
        array([[0.  , 0.  ],
               [0.25, 0.75],
               [0.5 , 0.5 ],
               [0.75, 0.25]])

        Generating vector from [https://github.com/QMCSoftware/LDData/tree/main/lattice](https://github.com/QMCSoftware/LDData/tree/main/lattice)

        >>> Lattice(dimension=3,randomize=False,generating_vector="mps.exod2_base2_m20_CKN.txt")(8,warn=False)
        array([[0.   , 0.   , 0.   ],
               [0.5  , 0.5  , 0.5  ],
               [0.25 , 0.75 , 0.75 ],
               [0.75 , 0.25 , 0.25 ],
               [0.125, 0.375, 0.375],
               [0.625, 0.875, 0.875],
               [0.375, 0.125, 0.125],
               [0.875, 0.625, 0.625]])

        Random generating vector supporting $2^{25}$ points

        >>> discrete_distrib = Lattice(3,generating_vector=25,seed=55,randomize=False)
        >>> discrete_distrib.gen_vec
        array([[       1, 11961679, 12107519]], dtype=uint64)
        >>> discrete_distrib(4,warn=False)
        array([[0.  , 0.  , 0.  ],
               [0.5 , 0.5 , 0.5 ],
               [0.25, 0.75, 0.75],
               [0.75, 0.25, 0.25]])

        Two random generating vectors both supporting $2^{25}$ points along with independent random shifts

        >>> discrete_distrib = Lattice(3,seed=7,generating_vector=25,replications=2)
        >>> discrete_distrib.gen_vec
        array([[       1, 32809149,  1471719],
               [       1,   275319, 19705657]], dtype=uint64)
        >>> discrete_distrib(4)
        array([[[0.3691824 , 0.65212985, 0.69669968],
                [0.8691824 , 0.15212985, 0.19669968],
                [0.6191824 , 0.90212985, 0.44669968],
                [0.1191824 , 0.40212985, 0.94669968]],
        <BLANKLINE>
               [[0.10605352, 0.63025643, 0.13630282],
                [0.60605352, 0.13025643, 0.63630282],
                [0.35605352, 0.38025643, 0.38630282],
                [0.85605352, 0.88025643, 0.88630282]]])

    **References**

    1.  Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama, Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou.
        GAIL: Guaranteed Automatic Integration Library (Version 2.3), MATLAB Software, 2019.
        [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).

    2.  F.Y. Kuo, D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients \- a survey of analysis and implementation.
        Foundations of Computational Mathematics, 16(6):1631-1696, 2016.
        [https://link.springer.com/article/10.1007/s10208-016-9329-5](https://link.springer.com/article/10.1007/s10208-016-9329-5).

    3.  D. Nuyens.
        The Magic Point Shop of QMC point generators and generating vectors.
        MATLAB and Python software, 2018.
        [https://people.cs.kuleuven.be/~dirk.nuyens/](https://people.cs.kuleuven.be/~dirk.nuyens/).

    4.  R. Cools, F.Y. Kuo, D. Nuyens.
        Constructing embedded lattice rules for multivariate integration.
        SIAM J. Sci. Comput., 28(6), 2162-2188.

    5.  P. L'Ecuyer, D. Munger.
        LatticeBuilder: A General Software Tool for Constructing Rank-1 Lattice Rules.
        ACM Transactions on Mathematical Software. 42. (2015).
        [10.1145/2754929](https://dl.acm.org/doi/10.1145/2754929).
    """

    def __init__(
        self,
        dimension=1,
        replications=None,
        seed=None,
        randomize="SHIFT",
        generating_vector="kuo.lattice-33002-1024-1048576.9125.txt",
        order="RADICAL INVERSE",
        m_max=None,
    ):
        r"""
        Args:
            dimension (Union[int, np.ndarray]): Dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1.
                - If an `np.ndarray` is passed in, use generating vector components at these indices.

            replications (int): Number of independent randomizations.
            seed (Union[None, int, np.random.SeedSeq): Seed the random number generator for reproducibility.
            randomize (str): Options are

                - `'SHIFT'`: Random shift.
                - `'FALSE'`: No randomization. In this case the first point will be the origin.

            generating_vector (Union[str, np.ndarray, int]): Specify the generating vector.

                - A `str` should be the name (or path) of a file from the LDData repo at [https://github.com/QMCSoftware/LDData/tree/main/lattice](https://github.com/QMCSoftware/LDData/tree/main/lattice).
                - A `np.ndarray` of integers with shape $(d,)$ or $(r,d)$ where $d$ is the number of dimensions and $r$ is the number of replications.
                    Must supply `m_max` where $2^{m_\mathrm{max}}$ is the max number of supported samples.
                - An `int`, call it $M$,
                gives the random generating vector $(1,v_1,\dots,v_{d-1})^T$
                where $d$ is the dimension and $v_i$ are randomly selected from $\{3,5,\dots,2^M-1\}$ uniformly and independently.
                We require require $1 < M < 27$.

            order (str): `'LINEAR'`, `'RADICAL INVERSE'`, or `'GRAY'` ordering. See the doctest example above.
            m_max (int): $2^{m_\mathrm{max}}$ is the maximum number of supported samples.
        """
        self.parameters = ["randomize", "gen_vec_source", "order", "n_limit"]
        self.input_generating_vector = deepcopy(generating_vector)
        self.input_m_max = deepcopy(m_max)
        if (
            isinstance(generating_vector, str)
            and generating_vector == "kuo.lattice-33002-1024-1048576.9125.txt"
        ):
            self.gen_vec_source = generating_vector
            gen_vec = np.load(
                dirname(abspath(__file__))
                + "/generating_vectors/kuo.lattice-33002-1024-1048576.9125.npy"
            )[None, :]
            d_limit = 9125
            n_limit = 1048576
        elif (
            isinstance(generating_vector, str)
            and generating_vector == "kuo.lattice-39102-1024-1048576.3600.txt"
        ):
            self.gen_vec_source = generating_vector
            gen_vec = np.load(
                dirname(abspath(__file__))
                + "/generating_vectors/kuo.lattice-39102-1024-1048576.3600.npy"
            )[None, :]
            d_limit = 3600
            n_limit = 1048576
        elif isinstance(generating_vector, str):
            self.gen_vec_source = generating_vector
            assert generating_vector[-4:] == ".txt"
            local_root = dirname(abspath(__file__)) + "/generating_vectors/"
            repos = DataSource()
            if repos.exists(local_root + generating_vector):
                datafile = repos.open(local_root + generating_vector)
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"
                + generating_vector
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"
                    + generating_vector
                )
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"
                + generating_vector
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"
                    + generating_vector
                )
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"
                + generating_vector[7:]
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/lattice/"
                    + generating_vector[7:]
                )
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/" + generating_vector
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/" + generating_vector
                )
            elif repos.exists(generating_vector):
                datafile = repos.open(generating_vector)
            else:
                raise ParameterError("LDData path %s not found" % generating_vector)
            contents = [
                int(line.rstrip("\n").strip().split("#", 1)[0])
                for line in datafile.readlines()
                if line[0] != "#"
            ]
            datafile.close()
            d_limit = int(contents[0])
            n_limit = int(contents[1])
            gen_vec = np.array(contents[2:], dtype=np.uint64)[None, :]
        elif isinstance(generating_vector, np.ndarray):
            self.gen_vec_source = "custom"
            gen_vec = generating_vector
            if m_max is None:
                raise ParameterError(
                    "m_max must be supplied when generating_vector is a np.ndarray"
                )
            n_limit = int(2**m_max)
            d_limit = int(gen_vec.shape[-1])
        elif isinstance(generating_vector, int):
            assert 1 < generating_vector < 27, "int generating vector out of range"
            n_limit = 2**generating_vector
            assert isinstance(
                dimension, int
            ), "random generating vector requires int dimension"
            d_limit = dimension
        else:
            raise ParameterError(
                "invalid generating_vector, must be a string, numpy.ndarray, or int"
            )
        super(Lattice, self).__init__(dimension, replications, seed, d_limit, n_limit)
        if isinstance(generating_vector, int):
            self.gen_vec_source = "random"
            m_max = int(np.log2(self.n_limit))
            gen_vec = np.hstack(
                [
                    np.ones((self.replications, 1), dtype=np.uint64),
                    2
                    * self.rng.integers(
                        1,
                        2 ** (m_max - 1),
                        size=(self.replications, dimension - 1),
                        dtype=np.uint64,
                    )
                    + 1,
                ]
            ).copy()
        assert isinstance(gen_vec, np.ndarray)
        gen_vec = np.atleast_2d(gen_vec)
        assert (
            gen_vec.ndim == 2
            and gen_vec.shape[1] >= self.d
            and (gen_vec.shape[0] == 1 or gen_vec.shape[0] == self.replications)
        ), "invalid gen_vec.shape = %s" % str(gen_vec.shape)
        self.gen_vec = gen_vec[:, self.dvec].copy()
        self.order = str(order).upper().strip().replace("_", " ")
        if self.order == "GRAY CODE":
            self.order = "GRAY"
        if self.order == "NATURAL":
            self.order = "RADICAL INVERSE"
        assert self.order in ["LINEAR", "RADICAL INVERSE", "GRAY"]
        self.randomize = str(randomize).upper()
        if self.randomize == "TRUE":
            self.randomize = "SHIFT"
        if self.randomize == "NONE":
            self.randomize = "FALSE"
        if self.randomize == "NO":
            self.randomize = "FALSE"
        assert self.randomize in ["SHIFT", "FALSE"]
        if self.randomize == "SHIFT":
            self.shift = self.rng.uniform(size=(self.replications, self.d))
        if self.randomize == "FALSE":
            assert self.gen_vec.shape[0] == self.replications, (
                "randomize='FALSE' but replications = %d does not equal the number of sets of generating vectors %d"
                % (self.replications, self.gen_vec.shape[0])
            )

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if return_binary:
            raise ParameterError("Lattice does not support return_binary=True")
        if n_min == 0 and self.randomize == "FALSE" and warn:
            warnings.warn(
                "Without randomization, the first lattice point is the origin",
                ParameterWarning,
            )
        r_x = np.uint64(self.gen_vec.shape[0])
        n = np.uint64(n_max - n_min)
        d = np.uint64(self.d)
        n_start = np.uint64(n_min)
        x = np.empty((r_x, n, d), dtype=np.float64)
        if self.order == "LINEAR":
            assert (
                r_x == 1
            ), "lattice linear currently requires there be only 1 generating matrix"
            x = self._gail_linear(n_min, n_max)[None, :, :]
        elif self.order == "RADICAL INVERSE":
            assert (n_min == 0 or (n_min & (n_min - 1)) == 0) and (
                n_max == 0 or (n_max & (n_max - 1)) == 0
            ), "lattice in natural order requires n_min and n_max be 0 or powers of 2"
            _ = qmctoolscl.lat_gen_natural(
                r_x, n, d, n_start, self.gen_vec, x, backend="c"
            )
        elif self.order == "GRAY":
            _ = qmctoolscl.lat_gen_gray(
                r_x, n, d, n_start, self.gen_vec, x, backend="c"
            )
        else:
            assert False, "invalid lattice order"
        if self.randomize == "FALSE":
            xr = x
        elif self.randomize == "SHIFT":
            r = np.uint64(self.replications)
            xr = np.empty((r, n, d), dtype=np.float64)
            qmctoolscl.lat_shift_mod_1(r, n, d, r_x, x, self.shift, xr, backend="c")
        return xr

    def _gen_block_linear(self, m_next, first=True):
        n = int(2**m_next)
        if first:
            y = np.arange(0, 1, 1 / n).reshape((n, 1))
        else:
            y = np.arange(1 / n, 1, 2 / n).reshape((n, 1))
        x = np.outer(y, self.gen_vec) % 1
        return x

    def calculate_y(self, m_low, m_high, y):
        for m in range(m_low, m_high):
            n = 2**m
            y_next = np.arange(1 / n, 1, 2 / n).reshape((int(n / 2), 1))
            temp = np.zeros((n, 1))
            temp[0::2] = y
            temp[1::2] = y_next
            y = temp
        return y

    def _gail_linear(self, n_min, n_max):
        """Gail lattice generator in linear order."""
        m_low = int(np.floor(np.log2(n_min))) + 1 if n_min > 0 else 0
        m_high = int(np.ceil(np.log2(n_max)))
        if n_min == 0:
            return self._gen_block_linear(m_high, first=True)
        else:
            n = 2 ** (m_low)
            y = np.arange(1 / n, 1, 2 / n).reshape((int(n / 2), 1))
            y = self.calculate_y(m_low, m_high, y)
            x = np.outer(y, self.gen_vec) % 1
            return x

    def _spawn(self, child_seed, dimension):
        return Lattice(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            randomize=self.randomize,
            generating_vector=self.input_generating_vector,
            order=self.order,
            m_max=self.input_m_max,
        )

    def expected_squared_periodic_discrepancies(self, n_max, coord_weights=None, kernel=None):
        """Returns the expected squared periodic discrepancies for each of the first n_max points of the lattice sequence.

        Args:
            n_max (int): Maximum number of points to calculate the squared periodic discrepancies for.
            coord_weights (Union[None, np.ndarray]): Coordinate weights for the discrepancy calculation. If None, uses weights gamma_j = j^(-2).
            kernel (Union[None, Callable]): Kernel function for the discrepancy calculation. If None, uses the second bernoulli polynomial.
        
        Returns:
            discs (np.ndarray): The expected squared periodic discrepancies for the first n_max points.
        """

        if coord_weights is not None and len(coord_weights) < self.d:
            raise ValueError("Length of coord_weights must be greater than or equal to the dimension of the lattice")
        if coord_weights is None:
            coord_weights = np.array([j**(-2) for j in range(1, self.d + 1)], dtype=np.float64)
        if self.order == "LINEAR":
            raise NotImplementedError("expected_squared_periodic_discrepancies not implemented for linear order")

        if kernel is None:
            kernel = lambda x: x * (x - 1) + 1/6

        coord_weights = coord_weights[:self.d]

        k_tilde = lambda x: np.prod(1 + coord_weights * kernel(x), axis=-1)
        
        # generate the vdc points without any random shift
        r_x = np.uint64(self.gen_vec.shape[0])
        n = np.uint64(2**(np.ceil(np.log2(n_max))))
        d = np.uint64(self.d)
        n_start = np.uint64(0)
        x = np.empty((r_x, n, d), dtype=np.float64)
        _ = qmctoolscl.lat_gen_natural(r_x, n, d, n_start, self.gen_vec, x, backend="c")
        s = x

        # evaluate the kernel on the sample points
        k_vector = k_tilde(s)
        k_vector = k_vector.reshape(-1)

        # get the constant vector term of the summation
        k_const = -1 + k_vector[0] / np.arange(1, n_max + 1, dtype=np.float64)

        # group the kernel evaluations by powers of 2
        k_sum = np.zeros(np.ceil(np.log2(n_max)).astype(int), dtype=np.float64)
        for i in range(k_sum.size):
            k_sum[i] = np.sum(k_vector[2**i:(2**(i+1))])

        # k_sum @ freq_mtx in O(n_max) memory: the shared 1/(j+1)**2 factor collapses the
        # (log2(n_max), n_max) frequency matrix to cumsum(2*w) / (j+1)**2, with w[j] the sum
        # of k_sum over the set bits of j.
        idx = np.arange(n_max)
        w = np.zeros(n_max, dtype=np.float64)
        for l in range(k_sum.size):
            w += k_sum[l] * ((idx >> l) & 1)
        discs = k_const + np.cumsum(2.0 * w) / (idx + 1.0) ** 2
        return discs


    def wssd(self, n_max, coord_weights=None, sample_weights=None, kernel=None):
        """Returns the weighted sum of the expected squared periodic discrepancies for the first n_max points of the lattice sequence.

        Args:
            n_max (int): Number of points to calculate the weighted squared periodic discrepancy for.
            coord_weights (Union[None, np.ndarray]): Coordinate weights for the discrepancy calculation. If None, uses weights gamma_j = j^(-2).
            sample_weights (Union[None, np.ndarray]): Sample weights for the weighted squared periodic discrepancy calculation. If None, uses weights w_n = n. Note that the time cost may be higher for other sample weights.
            kernel (Union[None, Callable]): Kernel function for the discrepancy calculation. If None, uses the second bernoulli polynomial.

        Returns:
            wssd (float): The weighted squared periodic discrepancy.
        """
        if coord_weights is not None and len(coord_weights) < self.d:
            raise ValueError("Length of coord_weights must be greater than or equal to the dimension of the lattice")
        if coord_weights is not None:
            coord_weights = coord_weights[:self.d]
        if sample_weights is not None and len(sample_weights) != n_max:
            raise ValueError("Length of sample_weights must equal n_max")
        if sample_weights is None:
            sample_weights = np.arange(1, n_max + 1, dtype=np.float64)

        discs = self.expected_squared_periodic_discrepancies(n_max, coord_weights=coord_weights, kernel=kernel)
        wssd = np.dot(sample_weights, discs)

        return wssd
