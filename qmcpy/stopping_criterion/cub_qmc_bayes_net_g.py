from .abstract_cub_bayes_ld_g import AbstractCubBayesLDG
from ..discrete_distribution import DigitalNetB2
from ..integrand import Keister, BoxIntegral, Genz, SensitivityIndices
from ..fast_transform import fwht, omega_fwht
from ..util import (
    MaxSamplesWarning,
    ParameterError,
    ParameterWarning,
    NotYetImplemented,
)
import ctypes
import numpy as np
from time import time
import warnings


class CubQMCBayesNetG(AbstractCubBayesLDG):
    r"""
    Quasi-Monte Carlo stopping criterion using fast Bayesian cubature and digital nets
    with guarantees for Gaussian processes having certain digitally shift invariant kernels.

    Examples:
        >>> k = Keister(DigitalNetB2(2, seed=123456789))
        >>> sc = CubQMCBayesNetG(k,abs_tol=5e-3)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.809
            comb_bound_low  1.807
            comb_bound_high 1.810
            comb_bound_diff 0.003
            comb_flags      1
            n_total         2^(10)
            n               2^(10)
            time_integrate  ...
        CubQMCBayesNetG (AbstractStoppingCriterion)
            abs_tol         0.005
            rel_tol         0
            n_init          2^(8)
            n_limit         2^(22)
            order           1
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            [0. 0.]
            variance        [0.5 0.5]
            standard_deviation [0.707 0.707]
            covariance      [[0.5 0. ]
                             [0.  0.5]]
            decomp_type     PCA
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               2^(1)
            replications    1
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         123456789

        Vector outputs

        >>> f = BoxIntegral(DigitalNetB2(3,seed=7),s=[-1,1])
        >>> abs_tol = 1e-2
        >>> sc = CubQMCBayesNetG(f,abs_tol=abs_tol,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.18750491, 0.96076395])
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [1.188 0.961]
            comb_bound_low  [1.18 0.96]
            comb_bound_high [1.195 0.962]
            comb_bound_diff [0.014 0.002]
            comb_flags      [ True  True]
            n_total         2^(11)
            n               [2048  256]
            time_integrate  ...
        CubQMCBayesNetG (AbstractStoppingCriterion)
            abs_tol         0.010
            rel_tol         0
            n_init          2^(8)
            n_limit         2^(22)
            order           1
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
        >>> sc = CubQMCBayesNetG(integrand,abs_tol=5e-2,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [[0.009 0.194 0.657]
                             [0.036 0.312 0.783]]
            comb_bound_low  [[0.007 0.178 0.636]
                             [0.033 0.297 0.762]]
            comb_bound_high [[0.012 0.209 0.679]
                             [0.038 0.327 0.804]]
            comb_bound_diff [[0.005 0.031 0.043]
                             [0.004 0.03  0.042]]
            comb_flags      [[ True  True  True]
                             [ True  True  True]]
            n_total         2^(8)
            n               [[[256 256 256]
                              [256 256 256]
                              [256 256 256]]
        <BLANKLINE>
                             [[256 256 256]
                              [256 256 256]
                              [256 256 256]]]
            time_integrate  ...
        CubQMCBayesNetG (AbstractStoppingCriterion)
            abs_tol         0.050
            rel_tol         0
            n_init          2^(8)
            n_limit         2^(22)
            order           1
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

    **References:**

    1.  Jagadeeswaran, Rathinavel, and Fred J. Hickernell.
        "Fast automatic Bayesian cubature using Sobol’sampling."
        Advances in Modeling and Simulation: Festschrift for Pierre L'Ecuyer.
        Springer International Publishing, 2022. 301-318.

    2.  Jagadeeswaran Rathinavel,
        Fast automatic Bayesian cubature using matching kernels and designs,
        PhD thesis, Illinois Institute of Technology, 2019.

    3.  Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        [http://gailgithub.github.io/GAIL_Dev/](http://gailgithub.github.io/GAIL_Dev/).
        [https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesNet_g.m](https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubBayesNet_g.m).
    """

    def __init__(
        self,
        integrand,
        abs_tol=1e-2,
        rel_tol=0,
        n_init=2**8,
        n_limit=2**22,
        error_fun="EITHER",
        alpha=0.01,
        errbd_type="MLE",
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
            alpha (np.ndarray): Uncertainty level in $(0,1)$.
            errbd_type (str): Options are

                - `'MLE'`: Marginal Log Likelihood.
                - `'GCV'`: Generalized Cross Validation.
                - `'FULL'`: Full Bayes.
        """
        super(CubQMCBayesNetG, self).__init__(
            integrand,
            ft=fwht,
            omega=omega_fwht,
            ptransform=None,
            allowed_distribs=[DigitalNetB2],
            kernel=self._shift_inv_kernel_digital,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            n_init=n_init,
            n_limit=n_limit,
            alpha=alpha,
            error_fun=error_fun,
            errbd_type=errbd_type,
        )
        self.order = 1  # Currently supports only order=1
        # private properties
        # Full Bayes - assumes m and s^2 as hyperparameters
        # GCV - Generalized cross validation
        self.kernType = 1  # Type-1:
        self._xfullundtype = np.uint64
        if self.discrete_distrib.order != "RADICAL INVERSE":
            raise ParameterError(
                "CubQMCNet_g requires DigitalNetB2 with 'RADICAL INVERSE' order"
            )

    # Digitally shift invariant kernel
    # C1 : first row of the covariance matrix
    # Lambda : eigen values of the covariance matrix
    # Lambda_ring = fwht(C1 - 1)

    def _shift_inv_kernel_digital(
        self, xun, order, a, avoid_cancel_error, kern_type, debug_enable
    ):
        kernel_func = CubQMCBayesNetG.BuildKernelFunc(order)
        const_mult = 1 / 10

        if avoid_cancel_error:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (vec_C1m1, C1_alt) = self.kernel_t(a * const_mult, kernel_func(xun))
            lambda_factor = max(abs(vec_C1m1))
            C1_alt = C1_alt / lambda_factor
            vec_C1m1 = vec_C1m1 / lambda_factor

            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda_ring = np.real(
                fwht(vec_C1m1.copy()) * np.sqrt(vec_C1m1.shape[-1])
            )

            vec_lambda = vec_lambda_ring.copy()
            vec_lambda[0] = vec_lambda_ring[0] + len(vec_lambda_ring) / lambda_factor

            if debug_enable:
                # eigenvalues must be real : Symmetric pos definite Kernel
                vec_lambda_direct = np.real(
                    np.array(fwht(C1_alt) * np.sqrt(C1_alt.shape[-1]), dtype=float)
                )  # Note: fwht output not normalized
                if sum(abs(vec_lambda_direct - vec_lambda)) > 1:
                    print("Possible error: check vec_lambda_ring computation")
        else:
            # direct approach to compute first row of the kernel Gram matrix
            vec_C1 = np.prod(1 + a * const_mult * kernel_func(xun), 2)
            # eigenvalues must be real : Symmetric pos definite Kernel
            vec_lambda = np.real(fwht(vec_C1) * np.sqrt(vec_C1.shape[-1]))
            vec_lambda_ring = 0
            lambda_factor = 1

        return vec_lambda, vec_lambda_ring, lambda_factor

    # Builds High order walsh kernel function
    @staticmethod
    def BuildKernelFunc(order):
        # a1 = @(x)(-np.floor(np.log2(x)))
        def a1(x):
            out = -np.floor(np.log2(x + np.finfo(float).eps))
            out[x == 0] = 0  # a1 is zero when x is zero
            return out

        # t1 = @(x)(2.^(-a1(x)))
        def t1(x):
            out = 2 ** (-a1(x))
            out[x == 0] = 0  # t1 is zero when x is zero
            return out

        # t2 = @(x)(2.^(-2*a1(x)))
        def t2(x):
            out = 2 ** (-2 * a1(x))
            out[x == 0] = 0  # t2 is zero when x is zero
            return out

        s1 = lambda x: (1 - 2 * x)
        s2 = lambda x: (1 / 3 - 2 * (1 - x) * x)
        ts2 = lambda x: ((1 - 5 * t1(x)) / 2 - (a1(x) - 2) * x)
        ts3 = lambda x: (
            (1 - 43 * t2(x)) / 18 + (5 * t1(x) - 1) * x + (a1(x) - 2) * x**2
        )

        if order == 1:
            kernFunc = lambda x: (
                6 * ((1 / 6) - 2 ** (np.floor(np.log2(x + np.finfo(float).eps)) - 1))
            )
        elif order == 2:
            omega2_1D = lambda x: (s1(x) + ts2(x))
            kernFunc = omega2_1D
        elif order == 3:
            omega3_1D = lambda x: (s1(x) + s2(x) + ts3(x))
            kernFunc = omega3_1D
        else:
            raise NotYetImplemented("cubBayesNet_g: kernel order not yet supported")

        return kernFunc
