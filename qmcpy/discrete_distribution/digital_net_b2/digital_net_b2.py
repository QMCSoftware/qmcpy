from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError, ParameterWarning
import qmctoolscl
from os.path import dirname, abspath
import numpy as np
from numpy.lib.npyio import DataSource
import warnings
from copy import deepcopy
import platform

class DigitalNetB2(AbstractLDDiscreteDistribution):
    r"""
    Low discrepancy digital net in base 2.

    Note:
        - Digital net sample sizes should be powers of $2$ e.g. $1$, $2$, $4$, $8$, $16$, $\dots$.
        - The first point of an unrandomized digital nets is the origin.
        - `Sobol` is an alias for `DigitalNetB2`.
        - To use higher order digital nets, either:

            - Pass in `generating_matrices` *without* interlacing and supply `alpha`>1 to apply interlacing, or
            - Pass in `generating_matrices` *with* interlacing and set `alpha=1` to avoid additional interlacing

            i.e. do *not* pass in interlaced `generating_matrices` and set `alpha>1`, this will apply additional interlacing.

    Examples:
        >>> discrete_distrib = DigitalNetB2(2,seed=7)
        >>> discrete_distrib(4)
        array([[0.72162356, 0.914955  ],
               [0.16345554, 0.42964856],
               [0.98676255, 0.03436384],
               [0.42956655, 0.55876342]])
        >>> discrete_distrib(1) # first point in the sequence
        array([[0.72162356, 0.914955  ]])
        >>> discrete_distrib
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               2^(1)
            replications    1
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7

        Replications of independent randomizations

        >>> x = DigitalNetB2(dimension=3,seed=7,replications=2)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.24653277, 0.1821862 , 0.74732591],
                [0.68152903, 0.66169442, 0.42891961],
                [0.48139855, 0.79818233, 0.08201287],
                [0.91541325, 0.29520621, 0.77495809]],
        <BLANKLINE>
               [[0.44876891, 0.85899604, 0.50549679],
                [0.53635924, 0.04353443, 0.33564946],
                [0.23214143, 0.29281506, 0.06841036],
                [0.75295715, 0.60241448, 0.76962976]]])

        Different orderings (avoid warnings that the first point is the origin)

        >>> DigitalNetB2(dimension=2,randomize=False,order="GRAY")(n_min=2,n_max=4,warn=False)
        array([[0.75, 0.25],
               [0.25, 0.75]])
        >>> DigitalNetB2(dimension=2,randomize=False,order="RADICAL INVERSE")(n_min=2,n_max=4,warn=False)
        array([[0.25, 0.75],
               [0.75, 0.25]])

        Generating matrices from [https://github.com/QMCSoftware/LDData/tree/main/dnet](https://github.com/QMCSoftware/LDData/tree/main/dnet)

        >>> DigitalNetB2(dimension=3,randomize=False,generating_matrices="mps.nx_s5_alpha2_m32.txt")(8,warn=False)
        array([[0.        , 0.        , 0.        ],
               [0.75841841, 0.45284834, 0.48844557],
               [0.57679828, 0.13226272, 0.10061957],
               [0.31858402, 0.32113875, 0.39369111],
               [0.90278927, 0.45867532, 0.01803333],
               [0.14542431, 0.02548793, 0.4749614 ],
               [0.45587539, 0.33081476, 0.11474426],
               [0.71318879, 0.15377192, 0.37629925]])

        All randomizations

        >>> DigitalNetB2(dimension=3,randomize='LMS DS',seed=5)(8)
        array([[0.69346401, 0.20118185, 0.64779396],
               [0.43998032, 0.90102467, 0.0936172 ],
               [0.86663563, 0.60910036, 0.26043276],
               [0.11327376, 0.30772653, 0.93959283],
               [0.62102883, 0.79169756, 0.77051637],
               [0.37451038, 0.1231324 , 0.46634012],
               [0.94785596, 0.38577413, 0.13377215],
               [0.20121617, 0.71843325, 0.56293458]])
        >>> DigitalNetB2(dimension=3,randomize='LMS',seed=5)(8,warn=False)
        array([[0.        , 0.        , 0.        ],
               [0.75446077, 0.83265937, 0.69584079],
               [0.42329494, 0.65793842, 0.90427279],
               [0.67763292, 0.48937304, 0.33344964],
               [0.18550714, 0.97332905, 0.3772791 ],
               [0.93104851, 0.17195496, 0.82311652],
               [0.26221346, 0.31742386, 0.53093284],
               [0.50787715, 0.5172669 , 0.2101083 ]])
        >>> DigitalNetB2(dimension=3,randomize='DS',seed=5)(8)
        array([[0.68383949, 0.04047995, 0.42903182],
               [0.18383949, 0.54047995, 0.92903182],
               [0.93383949, 0.79047995, 0.67903182],
               [0.43383949, 0.29047995, 0.17903182],
               [0.55883949, 0.66547995, 0.05403182],
               [0.05883949, 0.16547995, 0.55403182],
               [0.80883949, 0.41547995, 0.80403182],
               [0.30883949, 0.91547995, 0.30403182]])
        >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=5)(8)
        array([[0.33595486, 0.05834975, 0.30066401],
               [0.89110875, 0.84905188, 0.81833285],
               [0.06846074, 0.59997956, 0.67064205],
               [0.6693703 , 0.25824002, 0.10469644],
               [0.44586618, 0.99161977, 0.1873488 ],
               [0.84245267, 0.16445553, 0.56544372],
               [0.18546359, 0.44859876, 0.97389524],
               [0.61215442, 0.64341386, 0.44529863]])

        Higher order net without randomization

        >>> DigitalNetB2(dimension=3,randomize='FALSE',seed=7,alpha=2)(4,warn=False)
        array([[0.    , 0.    , 0.    ],
               [0.75  , 0.75  , 0.75  ],
               [0.4375, 0.9375, 0.1875],
               [0.6875, 0.1875, 0.9375]])


        Higher order nets with randomizations and replications

        >>> DigitalNetB2(dimension=3,randomize='LMS DS',seed=7,replications=2,alpha=2)(4,warn=False)
        array([[[0.42955149, 0.89149058, 0.43867111],
                [0.68701828, 0.07601148, 0.51312447],
                [0.10088033, 0.16293661, 0.25144138],
                [0.85846252, 0.87103178, 0.70041789]],
        <BLANKLINE>
               [[0.27151905, 0.42406763, 0.21917369],
                [0.55035224, 0.67864387, 0.90033876],
                [0.19356758, 0.57589964, 0.00347701],
                [0.97235125, 0.32168581, 0.86920948]]])
        >>> DigitalNetB2(dimension=3,randomize='LMS',seed=7,replications=2,alpha=2)(4,warn=False)
        array([[[0.        , 0.        , 0.        ],
                [0.75817062, 0.96603053, 0.94947625],
                [0.45367986, 0.80295638, 0.18778553],
                [0.71171791, 0.2295424 , 0.76175441]],
        <BLANKLINE>
               [[0.        , 0.        , 0.        ],
                [0.78664636, 0.75470215, 0.86876474],
                [0.45336727, 0.99953621, 0.22253579],
                [0.73996397, 0.24544824, 0.9008679 ]]])
        >>> DigitalNetB2(dimension=3,randomize='DS',seed=7,replications=2,alpha=2)(4)
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.79386058, 0.33727432, 0.6191824 ],
                [0.48136058, 0.39977432, 0.4316824 ],
                [0.73136058, 0.64977432, 0.6816824 ]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.40212985, 0.44669968, 0.85605352],
                [0.83962985, 0.25919968, 0.16855352],
                [0.08962985, 0.50919968, 0.91855352]]])
        >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=7,replications=2,alpha=2)(4)
        array([[[0.46368517, 0.03964427, 0.62172094],
                [0.7498683 , 0.76141348, 0.4243043 ],
                [0.01729754, 0.97968459, 0.65963223],
                [0.75365329, 0.1903774 , 0.34141493]],
        <BLANKLINE>
               [[0.52252547, 0.5679709 , 0.05949112],
                [0.27248656, 0.36488289, 0.81844058],
                [0.94219959, 0.39172304, 0.20285965],
                [0.19716391, 0.64741585, 0.92494554]]])

    **References:**

    1.  Marius Hofert and Christiane Lemieux.
        qrng: (Randomized) Quasi-Random Number Generators (2019).
        R package version 0.0-7.
        [https://CRAN.R-project.org/package=qrng](https://CRAN.R-project.org/package=qrng).

    2.  Faure, Henri, and Christiane Lemieux.
        Implementation of Irreducible Sobol' Sequences in Prime Power Bases.
        Mathematics and Computers in Simulation 161 (2019): 13-22. Crossref. Web.

    3.  F.Y. Kuo, D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients \- a survey of analysis and implementation.
        Foundations of Computational Mathematics, 16(6):1631-1696, 2016.
        [https://link.springer.com/article/10.1007/s10208-016-9329-5](https://link.springer.com/article/10.1007/s10208-016-9329-5).

    4.  D. Nuyens.
        The Magic Point Shop of QMC point generators and generating vectors.
        MATLAB and Python software, 2018.
        [https://people.cs.kuleuven.be/~dirk.nuyens/](https://people.cs.kuleuven.be/~dirk.nuyens/).

    5.  R. Cools, F.Y. Kuo, D. Nuyens.
        Constructing embedded lattice rules for multivariate integration.
        SIAM J. Sci. Comput., 28(6), 2162-2188.

    6.  I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman.
        Quasi-Random Sequence Generators.
        Keldysh Institute of Applied Mathematics.
        Russian Academy of Sciences, Moscow (1992).

    7.  Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011).
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 2011.
        [10.1002/wilm.10056](https://onlinelibrary.wiley.com/doi/abs/10.1002/wilm.10056).

    8.  Paul Bratley and Bennett L. Fox.
        Algorithm 659: Implementing Sobol's quasirandom sequence generator.
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88-100. 1988.
        [https://doi.org/10.1145/42288.214372](https://doi.org/10.1145/42288.214372).
    """

    def __init__(
        self,
        dimension=1,
        replications=None,
        seed=None,
        randomize="LMS DS",
        generating_matrices="joe_kuo.6.21201.txt",
        order="RADICAL INVERSE",
        t=63,
        alpha=1,
        msb=None,
        _verbose=False,
        # deprecated
        graycode=None,
        t_max=None,
        t_lms=None,
    ):
        r"""
        Args:
            dimension (Union[int, np.ndarray]): Dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1.
                - If an `np.ndarray` is passed in, use generating vector components at these indices.

            replications (int): Number of independent randomizations of a pointset.
            seed (Union[None, int, np.random.SeedSeq): Seed the random number generator for reproducibility.
            randomize (str): Options are

                - `'LMS DS'`: Linear matrix scramble with digital shift.
                - `'LMS'`: Linear matrix scramble only.
                - `'DS'`: Digital shift only.
                - `'NUS'`: Nested uniform scrambling. Also known as Owen scrambling.
                - `'FALSE'`: No randomization. In this case the first point will be the origin.

            generating_matrices (Union[str, np.ndarray, int]): Specify the generating matrices.

                - A `str` should be the name (or path) of a file from the LDData repo at [https://github.com/QMCSoftware/LDData/tree/main/dnet](https://github.com/QMCSoftware/LDData/tree/main/dnet).
                - An `np.ndarray` of integers with shape $(d,m_\mathrm{max})$ or $(r,d,m_\mathrm{max})$ where $d$ is the number of dimensions, $r$ is the number of replications, and $2^{m_\mathrm{max}}$ is the maximum number of supported points. Setting `msb=False` will flip the bits of ints in the generating matrices.

            order (str): `'RADICAL INVERSE'`, or `'GRAY'` ordering. See the doctest example above.
            t (int): Number of bits in integer represetation of points *after* randomization. The number of bits in the generating matrices is inferred based on the largest value.
            alpha (int): Interlacing factor for higher order nets.
                When `alpha`>1, interlacing is performed regardless of the generating matrices,
                i.e., for `alpha`>1 do *not* pass in generating matrices which are already interlaced.
                The Note for this class contains more info.
            msb (bool): Flag for Most Significant Bit (MSB) vs Least Significant Bit (LSB) integer representations in generating matrices. If `msb=False` (LSB order), then integers in generating matrices will be bit-reversed.
            _verbose (bool): If `True`, print linear matrix scrambling matrices.
        """
        if graycode is not None:
            order = "GRAY" if graycode else "RADICAL INVERSE"
            warnings.warn(
                "graycode argument deprecated, set order='GRAY' or order='RADICAL INVERSE' instead. Using order='%s'"
                % order,
                ParameterWarning,
            )
        if t_lms is not None:
            t = t_lms
            warnings.warn(
                "t_lms argument deprecated. Set t instead. Using t = %d" % t,
                ParameterWarning,
            )
        if t_max is not None:
            warnings.warn(
                "t_max is deprecated as it can be inferred from the generating matrices. Set t to change the number of bits after randomization.",
                ParameterWarning,
            )
        self.parameters = [
            "randomize",
            "gen_mats_source",
            "order",
            "t",
            "alpha",
            "n_limit",
        ]
        self.input_generating_matrices = deepcopy(generating_matrices)
        self.input_t = deepcopy(t)
        self.input_msb = deepcopy(msb)
        if (
            isinstance(generating_matrices, str)
            and generating_matrices == "joe_kuo.6.21201.txt"
        ):
            self.gen_mats_source = generating_matrices
            if np.isscalar(dimension) and dimension <= 1024 and alpha == 1:
                gen_mats = np.load(
                    dirname(abspath(__file__))
                    + "/generating_matrices/joe_kuo.6.1024.npy"
                )[None, :]
                d_limit = 1024
            else:
                gen_mats = np.load(
                    dirname(abspath(__file__))
                    + "/generating_matrices/joe_kuo.6.21201.npy"
                )[None, :]
                d_limit = 21201
            msb = True
            n_limit = 4294967296
            self._t_curr = 32
            compat_shift = self._t_curr - t if self._t_curr >= t else 0
            if compat_shift > 0:
                warnings.warn(
                    "Truncating ints in generating matrix to have t = %d bits." % t,
                    ParameterWarning,
                )
            gen_mats = gen_mats >> compat_shift
        elif isinstance(generating_matrices, str):
            self.gen_mats_source = generating_matrices
            assert generating_matrices[-4:] == ".txt"
            local_root = dirname(abspath(__file__)) + "/generating_matrices/"
            repos = DataSource()
            if repos.exists(local_root + generating_matrices):
                datafile = repos.open(local_root + generating_matrices)
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"
                + generating_matrices
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"
                    + generating_matrices
                )
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"
                + generating_matrices
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"
                    + generating_matrices
                )
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"
                + generating_matrices[7:]
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"
                    + generating_matrices[7:]
                )
            elif repos.exists(
                "https://raw.githubusercontent.com/QMCSoftware/" + generating_matrices
            ):
                datafile = repos.open(
                    "https://raw.githubusercontent.com/QMCSoftware/"
                    + generating_matrices
                )
            elif repos.exists(generating_matrices):
                datafile = repos.open(generating_matrices)
            else:
                raise ParameterError("LDData path %s not found" % generating_matrices)
            contents = [line.rstrip("\n").strip() for line in datafile.readlines()]
            contents = [line.split("#", 1)[0] for line in contents if line[0] != "#"]
            datafile.close()
            msb = True
            assert int(contents[0]) == 2, "DigitalNetB2 requires base=2 "  # base 2
            d_limit = int(contents[1])
            n_limit = int(contents[2])
            self._t_curr = int(contents[3])
            compat_shift = self._t_curr - t if self._t_curr >= t else 0
            if compat_shift > 0:
                warnings.warn(
                    "Truncating ints in generating matrix to have t = %d bits." % t,
                    ParameterWarning,
                )
            gen_mats = np.array(
                [
                    [int(v) >> compat_shift for v in line.split(" ")]
                    for line in contents[4:]
                ],
                dtype=np.uint64,
            )[None, :]
        elif isinstance(generating_matrices, np.ndarray):
            self.gen_mats_source = "custom"
            assert generating_matrices.ndim == 2 or generating_matrices.ndim == 3
            gen_mats = (
                generating_matrices[None, :, :]
                if generating_matrices.ndim == 2
                else generating_matrices
            )
            assert isinstance(
                msb, bool
            ), "when generating_matrices is a np.ndarray you must set either msb=True (for most significant bit ordering) or msb=False (for least significant bit ordering which will require a bit reversal)"
            gen_mat_max = gen_mats.max()
            assert gen_mat_max > 0, "generating matrix must have positive ints"
            self._t_curr = int(np.ceil(np.log2(gen_mat_max + 1)))
            d_limit = gen_mats.shape[1]
            n_limit = int(2 ** (gen_mats.shape[2]))
        else:
            raise ParameterError(
                "invalid generating_matrices, must be a string or np.ndarray."
            )
        super(DigitalNetB2, self).__init__(
            dimension, replications, seed, d_limit, n_limit
        )
        assert (
            gen_mats.ndim == 3
            and gen_mats.shape[1] >= self.d
            and (gen_mats.shape[0] == 1 or gen_mats.shape[0] == self.replications)
            and gen_mats.shape[2] > 0
        ), "invalid gen_mats.shape = %s" % str(gen_mats.shape)
        self.m_max = int(gen_mats.shape[-1])
        if isinstance(generating_matrices, np.ndarray) and (not msb):
            qmctoolscl.dnb2_gmat_lsb_to_msb(
                np.uint64(gen_mats.shape[0]),
                np.uint64(self.d),
                np.uint64(self.m_max),
                np.tile(np.uint64(self._t_curr), int(gen_mats.shape[0])),
                gen_mats,
                gen_mats,
                backend="c",
            )
        self.order = str(order).upper().strip().replace("_", " ")
        if self.order == "GRAY CODE":
            self.order = "GRAY"
        if self.order == "NATURAL":
            self.order = "RADICAL INVERSE"
        assert self.order in ["RADICAL INVERSE", "GRAY"]
        assert isinstance(t, int) and t > 0
        assert self._t_curr <= t <= 64, (
            "t must no more than 64 and no less than %d (the number of bits used to represent the generating matrices)"
            % (self._t_curr)
        )
        assert isinstance(alpha, int) and alpha > 0
        self.alpha = alpha
        if self.alpha > 1:
            assert (
                self.dvec == np.arange(self.d)
            ).all(), "digital interlacing requires dimension is an int"
            if self.m_max != self._t_curr:
                warnings.warn(
                    "Digital interlacing is often performed on matrices with the number of columns (m_max = %d) equal to the number of bits in each int (%d), but this is not the case. Ensure you are NOT setting alpha>1 when generating matrices are already interlaced."
                    % (self.m_max, self._t_curr),
                    ParameterWarning,
                )
        self._verbose = _verbose
        self.randomize = str(randomize).upper().strip().replace("_", " ")
        if self.randomize == "TRUE":
            self.randomize = "LMS DS"
        if self.randomize == "OWEN":
            self.randomize = "NUS"
        if self.randomize == "NONE":
            self.randomize = "FALSE"
        if self.randomize == "NO":
            self.randomize = "FALSE"
        assert self.randomize in ["LMS DS", "LMS", "DS", "NUS", "FALSE"]
        self.dtalpha = self.alpha * self.d
        if self.randomize == "FALSE":
            if self.alpha == 1:
                self.gen_mats = gen_mats[:, self.dvec, :]
                self.t = self._t_curr
            else:
                t_alpha = min(self.alpha * self._t_curr, t)
                gen_mat_ho = np.empty(
                    (gen_mats.shape[0], self.d, self.m_max), dtype=np.uint64
                )
                qmctoolscl.dnb2_interlace(
                    np.uint64(gen_mats.shape[0]),
                    np.uint64(self.d),
                    np.uint64(self.m_max),
                    np.uint64(self.dtalpha),
                    np.uint64(self._t_curr),
                    np.uint64(t_alpha),
                    np.uint64(self.alpha),
                    gen_mats[:, : self.dtalpha, :].copy(),
                    gen_mat_ho,
                    backend="c",
                )
                self.gen_mats = gen_mat_ho
                self._t_curr = t_alpha
                self.t = self._t_curr
        elif self.randomize == "DS":
            if self.alpha == 1:
                self.gen_mats = gen_mats[:, self.dvec, :]
                self.t = t
            else:
                t_alpha = min(self.alpha * self._t_curr, t)
                gen_mat_ho = np.empty(
                    (gen_mats.shape[0], self.d, self.m_max), dtype=np.uint64
                )
                qmctoolscl.dnb2_interlace(
                    np.uint64(gen_mats.shape[0]),
                    np.uint64(self.d),
                    np.uint64(self.m_max),
                    np.uint64(self.dtalpha),
                    np.uint64(self._t_curr),
                    np.uint64(t_alpha),
                    np.uint64(self.alpha),
                    gen_mats[:, : self.dtalpha, :].copy(),
                    gen_mat_ho,
                    backend="c",
                )
                self.gen_mats = gen_mat_ho
                self._t_curr = t_alpha
                self.t = t
            self.rshift = qmctoolscl.random_tbit_uint64s(
                self.rng, self.t, (self.replications, self.d)
            )
        elif self.randomize in ["LMS", "LMS DS"]:
            if self.alpha == 1:
                gen_mat_lms = np.empty(
                    (self.replications, self.d, self.m_max), dtype=np.uint64
                )
                S = qmctoolscl.dnb2_get_linear_scramble_matrix(
                    self.rng,
                    np.uint64(self.replications),
                    np.uint64(self.d),
                    np.uint64(self._t_curr),
                    np.uint64(t),
                    np.uint64(self._verbose),
                )
                qmctoolscl.dnb2_linear_matrix_scramble(
                    np.uint64(self.replications),
                    np.uint64(self.d),
                    np.uint64(self.m_max),
                    np.uint64(gen_mats.shape[0]),
                    np.uint64(t),
                    S,
                    gen_mats[:, self.dvec, :].copy(),
                    gen_mat_lms,
                    backend="c",
                )
                self.gen_mats = gen_mat_lms
                self._t_curr = t
                self.t = self._t_curr
            else:
                gen_mat_lms = np.empty(
                    (self.replications, self.dtalpha, self.m_max), dtype=np.uint64
                )
                S = qmctoolscl.dnb2_get_linear_scramble_matrix(
                    self.rng,
                    np.uint64(self.replications),
                    np.uint64(self.dtalpha),
                    np.uint64(self._t_curr),
                    np.uint64(t),
                    np.uint64(self._verbose),
                )
                qmctoolscl.dnb2_linear_matrix_scramble(
                    np.uint64(self.replications),
                    np.uint64(self.dtalpha),
                    np.uint64(self.m_max),
                    np.uint64(gen_mats.shape[0]),
                    np.uint64(t),
                    S,
                    gen_mats[:, : self.dtalpha, :].copy(),
                    gen_mat_lms,
                    backend="c",
                )
                gen_mat_lms_ho = np.empty(
                    (self.replications, self.d, self.m_max), dtype=np.uint64
                )
                qmctoolscl.dnb2_interlace(
                    np.uint64(self.replications),
                    np.uint64(self.d),
                    np.uint64(self.m_max),
                    np.uint64(self.dtalpha),
                    np.uint64(t),
                    np.uint64(t),
                    np.uint64(self.alpha),
                    gen_mat_lms,
                    gen_mat_lms_ho,
                    backend="c",
                )
                self.gen_mats = gen_mat_lms_ho
                self._t_curr = t
                self.t = self._t_curr
            if self.randomize == "LMS DS":
                self.rshift = qmctoolscl.random_tbit_uint64s(
                    self.rng, self.t, (self.replications, self.d)
                )
        elif self.randomize == "NUS":
            if alpha == 1:
                new_seeds = self._base_seed.spawn(self.replications * self.d)
                self.rngs = np.array(
                    [
                        np.random.Generator(np.random.SFC64(new_seeds[j]))
                        for j in range(self.replications * self.d)
                    ]
                ).reshape(self.replications, self.d)
                self.root_nodes = np.array(
                    [
                        qmctoolscl.NUSNode_dnb2()
                        for i in range(self.replications * self.d)
                    ]
                ).reshape(self.replications, self.d)
                self.gen_mats = gen_mats[:, self.dvec, :].copy()
                self.t = t
            else:
                self.dtalpha = self.alpha * self.d
                new_seeds = self._base_seed.spawn(self.replications * self.dtalpha)
                self.rngs = np.array(
                    [
                        np.random.Generator(np.random.SFC64(new_seeds[j]))
                        for j in range(self.replications * self.dtalpha)
                    ]
                ).reshape(self.replications, self.dtalpha)
                self.root_nodes = np.array(
                    [
                        qmctoolscl.NUSNode_dnb2()
                        for i in range(self.replications * self.dtalpha)
                    ]
                ).reshape(self.replications, self.dtalpha)
                self.gen_mats = gen_mats[:, : self.dtalpha, :].copy()
                self.t = t
        else:
            raise ParameterError("self.randomize parsing error")
        self.gen_mats = np.ascontiguousarray(self.gen_mats)
        gen_mat_max = self.gen_mats.max()
        assert gen_mat_max > 0, "generating matrix must have positive ints"
        assert self._t_curr == int(np.ceil(np.log2(gen_mat_max + 1)))
        assert (
            0 < self._t_curr <= self.t <= 64
        ), "invalid 0 <= self._t_curr (%d) <= self.t (%d) <= 64" % (
            self._t_curr,
            self.t,
        )
        if self.randomize == "FALSE":
            assert self.gen_mats.shape[0] == self.replications, (
                "randomize='FALSE' but replications = %d does not equal the number of sets of generating matrices %d"
                % (self.replications, self.gen_mats.shape[0])
            )

    def _try_gen_samples_float(self, r, n, d, n_start, mmax, r_x, return_binary):
        if return_binary or "NUS" in self.randomize:
            return None
        if self.order == "GRAY":
            gen_float = getattr(qmctoolscl, "dnb2_gen_gray_float", None)
        elif self.order == "RADICAL INVERSE":
            gen_float = getattr(qmctoolscl, "dnb2_gen_natural_float", None)
        else:
            return None
        if gen_float is None:
            return None
        x = np.empty((r, n, d), dtype=np.float64)
        tmaxes_new = np.tile(self.t, int(r)).astype(np.uint64)
        if "DS" in self.randomize:
            apply_shift = np.uint8(1)
            lshift = np.uint64(
                (self.t - self._t_curr) if "LMS" not in self.randomize else 0
            )
            lshifts = np.tile(lshift, int(r))
            shiftsb = self.rshift
        else:
            apply_shift = np.uint8(0)
            lshifts = np.zeros(max(1, int(r_x)), dtype=np.uint64)
            shiftsb = np.zeros((max(1, int(r)), int(d)), dtype=np.uint64)
        gen_float(
            r,
            n,
            d,
            n_start,
            mmax,
            r_x,
            apply_shift,
            lshifts,
            shiftsb,
            tmaxes_new,
            self.gen_mats,
            x,
            backend="c",
        )
        return x

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        if n_min == 0 and self.randomize in ["FALSE", "LMS"] and warn:
            warnings.warn(
                "Without randomization, the first digtial net point is the origin",
                ParameterWarning,
            )
        r_x = np.uint64(self.gen_mats.shape[0])
        n = np.uint64(n_max - n_min)
        d = (
            np.uint64(self.dtalpha)
            if self.randomize == "NUS" and self.alpha > 1
            else np.uint64(self.d)
        )
        n_start = np.uint64(n_min)
        mmax = np.uint64(self.m_max)
        if not ( (n_min == 0 or np.log2(n_min) % 1 == 0) and (n_max == 0 or np.log2(n_max) % 1 == 0)):
            if self.order == "GRAY":
                if warn:
                    warnings.warn("DigitalNetB2 in graycode order recommends n_min and n_max be 0 or powers of 2 at which the digital net achieves superior uniformity properties")
            elif self.order == "RADICAL INVERSE":
                raise ParameterError("DigitalNetB2 in radical inverse order requires n_min and n_max be 0 or powers of 2")
            else:
                raise ValueError("invalid digital net order")
        r = np.uint64(self.replications)
        x = self._try_gen_samples_float(r, n, d, n_start, mmax, r_x, return_binary)
        if x is not None:
            return x
        xb = np.empty((r_x, n, d), dtype=np.uint64)
        if self.order == "GRAY":
            qmctoolscl.dnb2_gen_gray(
                r_x, n, d, n_start, mmax, self.gen_mats, xb, backend="c"
            )
        elif self.order == "RADICAL INVERSE":
            qmctoolscl.dnb2_gen_natural(
                r_x, n, d, n_start, mmax, self.gen_mats, xb, backend="c"
            )
        else:
            raise ValueError("invalid digital net order")
        if "NUS" in self.randomize:
            if self.alpha == 1:
                xrb = np.empty((r, n, d), dtype=np.uint64)
                xb = np.ascontiguousarray(xb)
                qmctoolscl.dnb2_nested_uniform_scramble(
                    r,
                    n,
                    d,
                    r_x,
                    np.uint64(self._t_curr),
                    np.uint64(self.t),
                    self.rngs,
                    self.root_nodes,
                    xb,
                    xrb,
                )
                xb = xrb
            else:
                d = np.uint64(self.d)
                dtalpha = np.uint64(self.dtalpha)
                t_alpha = self.t
                alpha = np.uint64(self.alpha)
                xrb_lo = np.empty((r, n, self.dtalpha), dtype=np.uint64)
                _ = qmctoolscl.dnb2_nested_uniform_scramble(
                    r,
                    n,
                    dtalpha,
                    r_x,
                    np.uint64(self._t_curr),
                    np.uint64(t_alpha),
                    self.rngs,
                    self.root_nodes,
                    xb,
                    xrb_lo,
                )
                xrb_t = np.empty((r, d, n), dtype=np.uint64)
                xrb_lo_t = np.moveaxis(xrb_lo, [1, 2], [2, 1]).copy()
                qmctoolscl.dnb2_interlace(
                    r,
                    d,
                    n,
                    dtalpha,
                    np.uint64(t_alpha),
                    np.uint64(self.t),
                    alpha,
                    xrb_lo_t,
                    xrb_t,
                )
                xb = np.moveaxis(xrb_t, [1, 2], [2, 1]).copy()
        if "DS" in self.randomize:
            xrb = np.empty((r, n, d), dtype=np.uint64)
            lshifts = np.tile(
                np.uint64(
                    (self.t - self._t_curr) if "LMS" not in self.randomize else 0
                ),
                int(r),
            )
            qmctoolscl.dnb2_digital_shift(
                r, n, d, r_x, lshifts, xb, self.rshift, xrb, backend="c"
            )
            xb = xrb
        if return_binary:
            return xb
        else:
            x = np.empty((r, n, d), dtype=np.float64)
            tmaxes_new = np.tile(self.t, int(r)).astype(np.uint64)
            qmctoolscl.dnb2_integer_to_float(r, n, d, tmaxes_new, xb, x, backend="c")
            return x

    def _spawn(self, child_seed, dimension):
        return DigitalNetB2(
            dimension=dimension,
            replications=None if self.no_replications else self.replications,
            seed=child_seed,
            randomize=self.randomize,
            generating_matrices=self.input_generating_matrices,
            order=self.order,
            t=self.input_t,
            alpha=self.alpha,
            msb=self.input_msb,
            _verbose=False,
            # deprecated
            graycode=None,
            t_max=None,
            t_lms=None,
        )
