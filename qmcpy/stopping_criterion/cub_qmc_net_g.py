from .abstract_cub_qmc_ld_g import AbstractCubQMCLDG, _default_fudge
from ..fast_transform import fwht, omega_fwht
from ..util import ParameterError
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Uniform
from ..integrand import Keister, BoxIntegral, CustomFun
from ..integrand.genz import Genz
from ..integrand.sensitivity_indices import SensitivityIndices
import numpy as np


class CubQMCNetG(AbstractCubQMCLDG):
    r"""
    Quasi-Monte Carlo stopping criterion using digital net cubature
    with guarantees for cones of functions with a predictable decay in the Walsh coefficients.

    Examples:
        >>> k = Keister(DigitalNetB2(seed=7))
        >>> sc = CubQMCNetG(k,abs_tol=1e-3,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.38046669)
        >>> data
        Data (Data)
            solution        1.380
            comb_bound_low  1.380
            comb_bound_high 1.381
            comb_bound_diff 6.72e-04
            comb_flags      1
            n_total         2^(11)
            n               2^(11)
            time_integrate  ...
        CubQMCNetG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            0
            variance        2^(-1)
            standard_deviation 0.707
            covariance      2^(-1)
            decomp_type     PCA
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               1
            replications    1
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7

        Vector outputs

        >>> f = BoxIntegral(DigitalNetB2(3,seed=7),s=[-1,1])
        >>> abs_tol = 1e-3
        >>> sc = CubQMCNetG(f,abs_tol=abs_tol,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.19003352, 0.96068403])
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [1.19  0.961]
            comb_bound_low  [1.189 0.96 ]
            comb_bound_high [1.191 0.962]
            comb_bound_diff [0.001 0.002]
            comb_flags      [ True  True]
            n_total         2^(14)
            n               [16384  1024]
            time_integrate  ...
        CubQMCNetG (AbstractStoppingCriterion)
            abs_tol         0.001
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
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
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7
        >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        >>> true_value = np.array([sol3neg1,sol31])
        >>> assert (abs(true_value-solution)<abs_tol).all()

        Sensitivity indices

        >>> function = Genz(DigitalNetB2(3,seed=7))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubQMCNetG(integrand,abs_tol=5e-4,rel_tol=0,check_cone=True)
        >>> solution,data = sc.integrate()
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [[0.02  0.196 0.667]
                             [0.036 0.303 0.782]]
            comb_bound_low  [[0.019 0.195 0.667]
                             [0.035 0.303 0.781]]
            comb_bound_high [[0.02  0.196 0.667]
                             [0.036 0.303 0.782]]
            comb_bound_diff [[0.001 0.    0.001]
                             [0.001 0.001 0.001]]
            comb_flags      [[ True  True  True]
                             [ True  True  True]]
            n_total         2^(16)
            n               [[[16384 65536 65536]
                              [16384 65536 65536]
                              [16384 65536 65536]]
        <BLANKLINE>
                             [[ 2048 16384 32768]
                              [ 2048 16384 32768]
                              [ 2048 16384 32768]]]
            time_integrate  ...
        CubQMCNetG (AbstractStoppingCriterion)
            abs_tol         5.00e-04
            rel_tol         0
            n_init          2^(10)
            n_limit         2^(35)
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
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               3
            replications    1
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7

        Control Variates

        >>> dnb2 = DigitalNetB2(dimension=4,seed=7)
        >>> integrand = CustomFun(
        ...     true_measure = Uniform(dnb2),
        ...     g = lambda t: np.stack([
        ...         1*t[...,0]+2*t[...,0]**2+3*t[...,0]**3,
        ...         2*t[...,1]+3*t[...,1]**2+4*t[...,1]**3,
        ...         3*t[...,2]+4*t[...,2]**2+5*t[...,2]**3]),
        ...     dimension_indv = (3,))
        >>> control_variates = [
        ...     CustomFun(
        ...         true_measure = Uniform(dnb2),
        ...         g = lambda t: np.stack([t[...,0],t[...,1],t[...,2]],axis=0),
        ...         dimension_indv = (3,)),
        ...     CustomFun(
        ...         true_measure = Uniform(dnb2),
        ...         g = lambda t: np.stack([t[...,0]**2,t[...,1]**2,t[...,2]**2],axis=0),
        ...         dimension_indv = (3,))]
        >>> control_variate_means = np.array([[1/2,1/2,1/2],[1/3,1/3,1/3]])
        >>> true_value = np.array([23/12,3,49/12])
        >>> abs_tol = 1e-6
        >>> sc = CubQMCNetG(integrand,abs_tol=abs_tol,rel_tol=0,control_variates=control_variates,control_variate_means=control_variate_means,update_cv_coeffs=False)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.91666667, 3.        , 4.08333333])
        >>> data.n
        array([ 8192,  8192, 16384])
        >>> assert (np.abs(true_value-solution)<abs_tol).all()
        >>> sc = CubQMCNetG(integrand,abs_tol=abs_tol,rel_tol=0,control_variates=control_variates,control_variate_means=control_variate_means,update_cv_coeffs=True)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.91666667, 3.        , 4.08333333])
        >>> data.n
        array([16384, 16384, 16384])
        >>> assert (np.abs(true_value-solution)<abs_tol).all()

    **References:**

    1.  Hickernell, Fred J., and Lluís Antoni Jiménez Rugama.
        "Reliable adaptive cubature using digital sequences."
        Monte Carlo and Quasi-Monte Carlo Methods: MCQMC, Leuven, Belgium, April 2014.
        Springer International Publishing, 2016.

    2.  Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).
        [https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubSobol_g.m](https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubSobol_g.m).
    """

    def __init__(
        self,
        integrand,
        abs_tol=1e-2,
        rel_tol=0.0,
        n_init=2**10,
        n_limit=2**35,
        error_fun="EITHER",
        fudge=_default_fudge,
        check_cone=False,
        control_variates=None,
        control_variate_means=None,
        update_cv_coeffs=False,
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
            control_variates (list): Integrands to use as control variates, each with the same underlying discrete distribution instance.
            control_variate_means (np.ndarray): Means of each control variate.
            update_cv_coeffs (bool): If set to true, the control variate coefficients are recomputed at each iteration.
                Otherwise they are estimated once after the initial sampling and then fixed.
        """
        if control_variates is None:
            control_variates = []
        if control_variate_means is None:
            control_variate_means = []
        super(CubQMCNetG, self).__init__(
            integrand,
            abs_tol,
            rel_tol,
            n_init,
            n_limit,
            fudge,
            check_cone,
            control_variates,
            control_variate_means,
            update_cv_coeffs,
            ptransform="none",
            ft=fwht,
            omega=omega_fwht,
            allowed_distribs=[DigitalNetB2],
            cast_complex=False,
            error_fun=error_fun,
        )
        if self.discrete_distrib.order != "RADICAL INVERSE":
            raise ParameterError(
                "CubQMCNet_g requires DigitalNetB2 with 'RADICAL INVERSE' order"
            )
