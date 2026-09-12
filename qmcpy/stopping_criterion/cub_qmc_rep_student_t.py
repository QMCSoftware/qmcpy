from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data
from ..discrete_distribution import DigitalNetB2
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractLDDiscreteDistribution,
)
from ..integrand.keister import Keister
from ..integrand.box_integral import BoxIntegral
from ..integrand.sensitivity_indices import SensitivityIndices
from ..integrand.genz import Genz
from ..util import (
    MaxSamplesWarning,
    ParameterWarning,
    ParameterError,
)
import numpy as np
from scipy.stats import t
from time import time
import warnings


class CubQMCRepStudentT(AbstractStoppingCriterion):
    _RESUME_REQUIRED_FIELDS = ("xfull", "yfull", "n", "n_rep", "_ysums", "n_max")

    r"""
    Quasi-Monte Carlo stopping criterion based on Student's $t$-distribution for multiple replications.

    Examples:
        >>> k = Keister(DigitalNetB2(seed=7,replications=25))
        >>> sc = CubQMCRepStudentT(k,abs_tol=1e-3,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.3803927)
        >>> data
        Data (Data)
            solution        1.380
            comb_bound_low  1.380
            comb_bound_high 1.381
            comb_bound_diff 0.002
            comb_flags      1
            n_total         6400
            n               6400
            n_rep           2^(8)
            time_integrate  ...
        CubQMCRepStudentT (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         0.001
            rel_tol         0
            n_init          2^(8)
            n_limit         2^(30)
        Keister (AbstractIntegrand)
        Gaussian (AbstractTrueMeasure)
            mean            0
            variance        2^(-1)
            standard_deviation 0.707
            covariance      2^(-1)
            decomp_type     PCA
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               1
            replications    25
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7

        Vector outputs

        >>> f = BoxIntegral(DigitalNetB2(3,seed=7,replications=25),s=[-1,1])
        >>> abs_tol = 1e-3
        >>> sc = CubQMCRepStudentT(f,abs_tol=abs_tol,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.19025707, 0.96062762])
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [1.19  0.961]
            comb_bound_low  [1.19  0.961]
            comb_bound_high [1.191 0.961]
            comb_bound_diff [0.001 0.   ]
            comb_flags      [ True  True]
            n_total         409600
            n               [409600   6400]
            n_rep           [16384   256]
            time_integrate  ...
        CubQMCRepStudentT (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         0.001
            rel_tol         0
            n_init          2^(8)
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
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               3
            replications    25
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

        >>> function = Genz(DigitalNetB2(3,seed=7,replications=25))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubQMCRepStudentT(integrand,abs_tol=5e-4,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [[0.02  0.196 0.667]
                             [0.036 0.303 0.782]]
            comb_bound_low  [[0.019 0.195 0.667]
                             [0.035 0.303 0.781]]
            comb_bound_high [[0.02  0.196 0.667]
                             [0.036 0.303 0.782]]
            comb_bound_diff [[0.001 0.001 0.001]
                             [0.001 0.001 0.001]]
            comb_flags      [[ True  True  True]
                             [ True  True  True]]
            n_total         204800
            n               [[[102400 204800 204800]
                              [102400 204800 204800]
                              [102400 204800 204800]]
        <BLANKLINE>
                             [[ 12800 102400 102400]
                              [ 12800 102400 102400]
                              [ 12800 102400 102400]]]
            n_rep           [[[4096 8192 8192]
                              [4096 8192 8192]
                              [4096 8192 8192]]
        <BLANKLINE>
                             [[ 512 4096 4096]
                              [ 512 4096 4096]
                              [ 512 4096 4096]]]
            time_integrate  ...
        CubQMCRepStudentT (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         5.00e-04
            rel_tol         0
            n_init          2^(8)
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
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               3
            replications    25
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7


    **References:**

    1.  Art B. Owen. "Practical Quasi-Monte Carlo Integration." 2023.
        [https://artowen.su.domains/mc/](https://artowen.su.domains/mc/).

    2.  Pierre l’Ecuyer et al.
        "Confidence intervals for randomized quasi-Monte Carlo estimators."
        2023 Winter Simulation Conference (WSC). IEEE, 2023.
        [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10408613](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10408613).
    """

    def __init__(
        self,
        integrand,
        abs_tol=1e-2,
        rel_tol=0.0,
        n_init=256.0,
        n_limit=2**30,
        error_fun="EITHER",
        inflate=1,
        alpha=0.01,
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
            inflate (float): Inflation factor $\geq 1$ to multiply by the variance estimate to make it more conservative.
            alpha (np.ndarray): Uncertainty level in $(0,1)$.
        """
        self.parameters = ["inflate", "alpha", "abs_tol", "rel_tol", "n_init", "n_limit"]
        # Input Checks
        if np.log2(n_init) % 1 != 0:
            warnings.warn(
                "n_init must be a power of two. Using n_init = 2**5", ParameterWarning
            )
            n_init = 2**5
        if np.log2(n_limit) % 1 != 0:
            warnings.warn(
                "n_init must be a power of two. Using n_limit = 2**30", ParameterWarning
            )
            n_limit = 2**30
        # Set Attributes
        self.n_init = int(n_init)
        self.n_limit = int(n_limit)
        assert isinstance(error_fun, str) or callable(error_fun)
        if isinstance(error_fun, str):
            if error_fun.upper() == "EITHER":
                error_fun = lambda sv, abs_tol, rel_tol: np.maximum(
                    abs_tol, abs(sv) * rel_tol
                )
            elif error_fun.upper() == "BOTH":
                error_fun = lambda sv, abs_tol, rel_tol: np.minimum(
                    abs_tol, abs(sv) * rel_tol
                )
            else:
                raise ParameterError("str error_fun must be 'EITHER' or 'BOTH'")
        self.error_fun = error_fun
        self.alpha = alpha
        self.inflate = float(inflate)
        assert self.inflate >= 1
        assert 0 < self.alpha < 1
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubQMCRepStudentT, self).__init__(
            allowed_distribs=[AbstractLDDiscreteDistribution],
            allow_vectorized_integrals=True,
        )
        assert (
            self.integrand.discrete_distrib.replications > 1
        ), "Require the discrete distribution has replications>1"
        assert (
            self.integrand.discrete_distrib.randomize != "FALSE"
        ), "Require discrete distribution is randomized"
        self.alphas_indv, _ = self._compute_indv_alphas(
            np.full(self.integrand.d_comb, self.alpha)
        )
        self.set_tolerance(abs_tol, rel_tol)
        self.t_star = -t.ppf(
            self.alphas_indv / 2, df=self.integrand.discrete_distrib.replications - 1
        )

    def integrate(self, resume=None):
        t_start = time()
        resume_provenance = self._capture_resume_provenance(resume)
        trace = self._make_trace_logger()
        data = self._prepare_resume_data(
            resume, self._validate_resume, self._restore_resume_state
        )
        if data is not None:
            # Reset flags so the tighter tolerance is re-evaluated from existing samples.
            data.flags_indv = np.tile(False, self.integrand.d_indv)
            data.compute_flags = np.tile(True, self.integrand.d_indv)
            self._set_elapsed_time(data, 0.0, resume_provenance=resume_provenance)
            trace.resume(data)
        else:
            data = Data(parameters=["solution", "comb_bound_low", "comb_bound_high", "comb_bound_diff", "comb_flags", "n_total", "n", "n_rep", "time_integrate"])
            data.flags_indv = np.tile(False, self.integrand.d_indv)
            data.compute_flags = np.tile(True, self.integrand.d_indv)
            data.n_rep = np.tile(self.n_init, self.integrand.d_indv)
            data.n_min = 0
            data.n_max = self.n_init
            data.solution_indv = np.tile(np.nan, self.integrand.d_indv)
            data.xfull = np.empty((self.discrete_distrib.replications, 0, self.integrand.d))
            data.yfull = np.empty(self.integrand.d_indv + (self.discrete_distrib.replications, 0))
            data._ysums = np.zeros(self.integrand.d_indv + (self.discrete_distrib.replications,), dtype=float)
        first_resume_iter = resume is not None
        while True:
            if not first_resume_iter:
                xnext = self.discrete_distrib(n_min=data.n_min, n_max=data.n_max)
                data.xfull = np.concatenate([data.xfull, xnext], 1)
                ynext = self.integrand.f(xnext, compute_flags=data.compute_flags)
                ynext[~data.compute_flags] = np.nan
                data.yfull = np.concatenate([data.yfull, ynext], -1)
                data.n_rep[data.compute_flags] = data.n_max
                data._ysums[data.compute_flags] += ynext[data.compute_flags].sum(-1)
            data.muhats = data._ysums / data.n_rep[..., None]
            data.solution_indv = data.muhats.mean(-1)
            data.sigmahat = data.muhats.std(-1, ddof=1)
            data.ci_half_width = self.t_star * self.inflate * data.sigmahat / np.sqrt(self.discrete_distrib.replications)
            data.indv_bound_low = data.solution_indv - data.ci_half_width
            data.indv_bound_high = data.solution_indv + data.ci_half_width
            data.n = self.discrete_distrib.replications * data.n_rep
            data.n_total = data.n.max()
            data.comb_bound_low, data.comb_bound_high = self.integrand.bound_fun(
                data.indv_bound_low, data.indv_bound_high
            )
            data.comb_bound_diff = data.comb_bound_high - data.comb_bound_low
            fidxs = np.isfinite(data.comb_bound_low) & np.isfinite(data.comb_bound_high)
            slow, shigh, abs_tols, rel_tols = (
                data.comb_bound_low[fidxs],
                data.comb_bound_high[fidxs],
                self.abs_tols[fidxs],
                self.rel_tols[fidxs],
            )
            data.solution = np.tile(np.nan, data.comb_bound_low.shape)
            data.solution[fidxs] = 0.5 * (slow + shigh + self.error_fun(slow, abs_tols, rel_tols) - self.error_fun(shigh, abs_tols, rel_tols))
            data.comb_flags = np.tile(False, data.comb_bound_low.shape)
            data.comb_flags[fidxs] = (shigh - slow) <= (
                self.error_fun(slow, abs_tols, rel_tols)
                + self.error_fun(shigh, abs_tols, rel_tols)
            )
            data.flags_indv = self.integrand.dependency(data.comb_flags)
            data.compute_flags = ~data.flags_indv
            self._set_elapsed_time(data, time() - t_start, resume_provenance=resume_provenance)
            trace.iteration(data)
            if np.sum(data.compute_flags) == 0:
                break  # sufficiently estimated
            elif 2 * data.n_total > self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_limit = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ % (int(data.n_total), int(data.n_total), int(self.n_limit))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            first_resume_iter = False
            data.n_min = data.n_max
            data.n_max = 2 * data.n_min
        self._finalize_integration_data(
            data, time() - t_start, resume_provenance=resume_provenance
        )
        trace.finalize()
        return data.solution, data

    def _validate_resume(self, data):
        self._validate_resume_data(data, required_fields=self._RESUME_REQUIRED_FIELDS)
        if int(data.stopping_crit.n_init) != int(self.n_init):
            raise ParameterError("resume data has incompatible n_init.")
        replications = int(self.discrete_distrib.replications)
        n_rep_max = int(np.max(np.asarray(data.n_rep)))
        if not self._is_power_of_two(n_rep_max):
            raise ParameterError("resume data n_rep must be a power of 2.")
        if int(data.n_max) != n_rep_max:
            raise ParameterError("resume data n_max must match max(n_rep).")
        if n_rep_max < int(self.n_init):
            raise ParameterError(
                "resume data must include at least n_init samples per replication."
            )
        if data.xfull.shape != (replications, n_rep_max, self.integrand.d):
            raise ParameterError(
                "resume data xfull shape must be (%d, %d, %d); got %s."
                % (replications, n_rep_max, self.integrand.d, data.xfull.shape)
            )
        expected_y_shape = self.integrand.d_indv + (replications, n_rep_max)
        if np.shape(data.yfull) != expected_y_shape:
            raise ParameterError(
                "resume data yfull shape must be %s; got %s."
                % (expected_y_shape, np.shape(data.yfull))
            )
        if np.shape(data._ysums) != self.integrand.d_indv + (replications,):
            raise ParameterError(
                "resume data _ysums shape must be %s; got %s."
                % (self.integrand.d_indv + (replications,), np.shape(data._ysums))
            )
        if np.shape(data.n) != self.integrand.d_indv:
            raise ParameterError("resume data n shape must match integrand.d_indv.")
        if np.shape(data.n_rep) != self.integrand.d_indv:
            raise ParameterError(
                "resume data n_rep shape must match integrand.d_indv."
            )
        if not np.array_equal(
            np.asarray(data.n), replications * np.asarray(data.n_rep)
        ):
            raise ParameterError(
                "resume data n must equal replications * n_rep componentwise."
            )
        if int(data.n_total) != replications * n_rep_max:
            raise ParameterError(
                "resume data n_total must equal replications * max(n_rep)."
            )

    def _restore_resume_state(self, data):
        self._restore_resume_rng_state(data)
        self.true_measure.discrete_distrib = self.discrete_distrib
        self.integrand.discrete_distrib = self.discrete_distrib
        self.integrand.true_measure.discrete_distrib = self.discrete_distrib

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rmse_tol is None, "rmse_tol not supported by this stopping criterion."
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb, self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb, self.rel_tol)
