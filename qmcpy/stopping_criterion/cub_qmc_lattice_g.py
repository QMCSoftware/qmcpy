from .abstract_cub_qmc_ld_g import AbstractCubQMCLDG, _default_fudge
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian, Uniform
from ..integrand import Keister, BoxIntegral, CustomFun
from ..integrand.genz import Genz
from ..integrand.sensitivity_indices import SensitivityIndices
from ..fast_transform import fftbr, omega_fftbr
from ..util import ParameterError
import numpy as np


class CubQMCLatticeG(AbstractCubQMCLDG):
    r"""
    Quasi-Monte Carlo stopping criterion using rank-1 lattice cubature
    with guarantees for cones of functions with a predictable decay in the Fourier coefficients.

    Examples:
        >>> k = Keister(Lattice(seed=7))
        >>> sc = CubQMCLatticeG(k,abs_tol=1e-3,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.38037385)
        >>> data
        Data (Data)
            solution        1.380
            comb_bound_low  1.380
            comb_bound_high 1.381
            comb_bound_diff 0.001
            comb_flags      1
            n_total         2^(11)
            n               2^(11)
            time_integrate  ...
        CubQMCLatticeG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            0
            variance        2^(-1)
            standard_deviation 0.707
            covariance      2^(-1)
            decomp_type     PCA
        Lattice (AbstractLDDiscreteDistribution)
            d               1
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           RADICAL INVERSE
            n_limit         2^(20)
            entropy         7

        Vector outputs

        >>> f = BoxIntegral(Lattice(3,seed=11),s=[-1,1])
        >>> abs_tol = 1e-3
        >>> sc = CubQMCLatticeG(f,abs_tol=abs_tol,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.18947477, 0.96060862])
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [1.189 0.961]
            comb_bound_low  [1.189 0.96 ]
            comb_bound_high [1.19  0.961]
            comb_bound_diff [0.001 0.001]
            comb_flags      [ True  True]
            n_total         2^(13)
            n               [8192 1024]
            time_integrate  ...
        CubQMCLatticeG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
        BoxIntegral (AbstractIntegrand)
            s               [-1  1]
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
            mean            [0.5 0.5 0.5]
            variance        [0.083 0.083 0.083]
            standard_deviation [0.289 0.289 0.289]
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                with 3 stored elements (1 diagonals) and shape (3, 3)>
                 Coords Values
                 (0, 0) 0.08333333333333333
                 (1, 1) 0.08333333333333333
                 (2, 2) 0.08333333333333333
        Lattice (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           RADICAL INVERSE
            n_limit         2^(20)
            entropy         11
        >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        >>> true_value = np.array([sol3neg1,sol31])
        >>> assert (abs(true_value-solution)<abs_tol).all()

        Sensitivity indices

        >>> function = Genz(Lattice(3,seed=7))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubQMCLatticeG(integrand,abs_tol=5e-4,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [[0.021 0.196 0.667]
                             [0.036 0.303 0.782]]
            comb_bound_low  [[0.02  0.196 0.667]
                             [0.035 0.303 0.781]]
            comb_bound_high [[0.021 0.196 0.667]
                             [0.036 0.303 0.782]]
            comb_bound_diff [[0.001 0.001 0.   ]
                             [0.001 0.001 0.001]]
            comb_flags      [[ True  True  True]
                             [ True  True  True]]
            n_total         2^(16)
            n               [[[16384 32768 65536]
                              [16384 32768 65536]
                              [16384 32768 65536]]
        <BLANKLINE>
                             [[ 2048 16384 32768]
                              [ 2048 16384 32768]
                              [ 2048 16384 32768]]]
            time_integrate  ...
        CubQMCLatticeG (AbstractStoppingCriterion)
            abs_tol         5.00e-04
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(30)
        SensitivityIndices (AbstractIntegrand)
            indices         [[ True False False]
                             [False  True False]
                             [False False  True]]
        Uniform (AbstractTrueMeasure)
            lower_bound     0
            upper_bound     1
            mean            [0.5 0.5 0.5]
            variance        [0.083 0.083 0.083]
            standard_deviation [0.289 0.289 0.289]
            covariance      <DIAgonal sparse matrix of dtype 'float64'
                with 3 stored elements (1 diagonals) and shape (3, 3)>
                 Coords Values
                 (0, 0) 0.08333333333333333
                 (1, 1) 0.08333333333333333
                 (2, 2) 0.08333333333333333
        Lattice (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       SHIFT
            gen_vec_source  kuo.lattice-33002-1024-1048576.9125.txt
            order           RADICAL INVERSE
            n_limit         2^(20)
            entropy         7

    **References:**

    1.  Lluis Antoni Jimenez Rugama and Fred J. Hickernell.
        "Adaptive multidimensional integration based on rank-1 lattices,"
        Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Leuven, Belgium,
        April 2014 (R. Cools and D. Nuyens, eds.), Springer Proceedings in Mathematics.
        and Statistics, vol. 163, Springer-Verlag, Berlin, 2016, arXiv:1411.1966, pp. 407-422.

    2.  Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).
        [https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubLattice_g.m](https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubLattice_g.m).
    """

    def __init__(
        self,
        integrand,
        abs_tol=1e-2,
        rel_tol=0.0,
        n_init=2**10,
        n_limit=2**30,
        error_fun="EITHER",
        fudge=_default_fudge,
        check_cone=False,
        ptransform="BAKER",
    ):
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rel_tol (np.ndarray): Relative error tolerance.
            n_init (int): Initial number of samples.
            n_limit (int): Maximum number of samples.
            error_fun (Union[str, callable]): Function mapping the approximate solution, absolute error tolerance, and relative error tolerance to the current error bound.

                - `'EITHER'`, the default, requires the approximation error must be below either the absolue *or* relative tolerance.
                    Equivalent to setting
                    ```python
                    error_fun = lambda sv,abs_tol,rel_tol: np.maximum(abs_tol,abs(sv)*rel_tol)
                    ```
                - `'BOTH'` requires the approximation error to be below both the absolue *and* relative tolerance.
                    Equivalent to setting
                    ```python
                    error_fun = lambda sv,abs_tol,rel_tol: np.minimum(abs_tol,abs(sv)*rel_tol)
                    ```
            fudge (function): Positive function multiplying the finite sum of the Fourier coefficients specified in the cone of functions.
            check_cone (bool): Whether or not to check if the function falls in the cone.
            ptransform (str): Periodization transform, see the options in `AbstractIntegrand.f`.
        """
        super(CubQMCLatticeG, self).__init__(
            integrand,
            abs_tol,
            rel_tol,
            n_init,
            n_limit,
            fudge,
            check_cone,
            control_variates=[],
            control_variate_means=[],
            update_beta=False,
            ptransform=ptransform,
            ft=fftbr,
            omega=omega_fftbr,
            allowed_distribs=[Lattice],
            cast_complex=True,
            error_fun=error_fun,
        )
        if self.discrete_distrib.order != "RADICAL INVERSE":
            raise ParameterError(
                "CubLattice_g requires Lattice with 'RADICAL INVERSE' order"
            )
