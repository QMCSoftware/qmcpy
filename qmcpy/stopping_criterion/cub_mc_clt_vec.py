from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data

from ..discrete_distribution import IIDStdUniform  #pylint: disable=unused-import
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractIIDDiscreteDistribution,
)
from ..integrand import Keister, BoxIntegral, CustomFun, Genz, SensitivityIndices  #pylint: disable=unused-import
from ..util import MaxSamplesWarning, ParameterWarning, ParameterError
import numpy as np
from time import time
from scipy.stats import norm
import warnings


class CubMCCLTVec(AbstractStoppingCriterion):
    r"""
    IID Monte Carlo stopping criterion stopping criterion based on the Central Limit Theorem with doubling sample sizes.

    Examples:
        >>> k = Keister(IIDStdUniform(seed=7))
        >>> sc = CubMCCLTVec(k,abs_tol=5e-2,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> solution
        array(1.38366791)
        >>> data
        Data (Data)
            solution        1.384
            comb_bound_low  1.343
            comb_bound_high 1.424
            comb_bound_diff 0.080
            comb_flags      1
            n_total         1024
            n               2^(10)
            time_integrate  ...
        CubMCCLTVec (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         0.050
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
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               1
            replications    1
            entropy         7

        Vector outputs

        >>> f = BoxIntegral(IIDStdUniform(3,seed=7),s=[-1,1])
        >>> abs_tol = 2.5e-2
        >>> sc = CubMCCLTVec(f,abs_tol=abs_tol,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> solution
        array([1.18448043, 0.95435347])
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [1.184 0.954]
            comb_bound_low  [1.165 0.932]
            comb_bound_high [1.203 0.977]
            comb_bound_diff [0.038 0.045]
            comb_flags      [ True  True]
            n_total         8192
            n               [8192 1024]
            time_integrate  ...
        CubMCCLTVec (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         0.025
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
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               3
            replications    1
            entropy         7
        >>> sol3neg1 = -np.pi/4-1/2*np.log(2)+np.log(5+3*np.sqrt(3))
        >>> sol31 = np.sqrt(3)/4+1/2*np.log(2+np.sqrt(3))-np.pi/24
        >>> true_value = np.array([sol3neg1,sol31])
        >>> assert (abs(true_value-solution)<abs_tol).all()

        Sensitivity indices

        >>> function = Genz(IIDStdUniform(3,seed=7))
        >>> integrand = SensitivityIndices(function)
        >>> sc = CubMCCLTVec(integrand,abs_tol=2.5e-2,rel_tol=0)
        >>> solution,data = sc.integrate()
        >>> data  # doctest: +NORMALIZE_WHITESPACE
        Data (Data)
            solution        [[0.024 0.203 0.662]
                             [0.044 0.308 0.78 ]]
            comb_bound_low  [[0.006 0.186 0.644]
                             [0.02  0.286 0.761]]
            comb_bound_high [[0.042 0.221 0.681]
                             [0.067 0.329 0.798]]
            comb_bound_diff [[0.036 0.035 0.037]
                             [0.047 0.043 0.037]]
            comb_flags      [[ True  True  True]
                             [ True  True  True]]
            n_total         262144
            n               [[[  4096  65536 262144]
                              [  4096  65536 262144]
                              [  4096  65536 262144]]
        <BLANKLINE>
                             [[   512  32768 262144]
                              [   512  32768 262144]
                              [   512  32768 262144]]]
            time_integrate  ...
        CubMCCLTVec (AbstractStoppingCriterion)
            inflate         1
            alpha           0.010
            abs_tol         0.025
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
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               3
            replications    1
            entropy         7
    """
    _RESUME_REQUIRED_FIELDS = ("xfull", "yfull", "n", "n_max")

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
        self.parameters = [
            "inflate",
            "alpha",
            "abs_tol",
            "rel_tol",
            "n_init",
            "n_limit",
        ]
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
        self.error_fun, _ = self._resolve_error_fun(error_fun)
        self.alpha = alpha
        self.inflate = float(inflate)
        assert self.inflate >= 1
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMCCLTVec, self).__init__(
            allowed_distribs=[AbstractIIDDiscreteDistribution],
            allow_vectorized_integrals=True,
        )
        assert (
            self.integrand.discrete_distrib.no_replications == True
        ), "Require the discrete distribution has replications=None"
        self.alphas_indv, _ = self._compute_indv_alphas(
            np.full(self.integrand.d_comb, self.alpha)
        )
        self.set_tolerance(abs_tol, rel_tol)
        self.z_star = -norm.ppf(self.alphas_indv / 2)

    def _validate_resume(self, data):
        self._validate_resume_data(data, required_fields=self._RESUME_REQUIRED_FIELDS)
        if int(data.stopping_crit.n_init) != int(self.n_init):
            raise ParameterError("resume data has incompatible n_init.")
        if data.xfull.shape[0] != int(data.n_total):
            raise ParameterError("resume data xfull length must match n_total.")
        if data.yfull.shape[-1] != int(data.n_total):
            raise ParameterError("resume data yfull length must match n_total.")
        if np.shape(data.n) != self.integrand.d_indv:
            raise ParameterError(
                "resume data n shape must match integrand.d_indv."
            )
        if int(np.max(np.asarray(data.n))) != int(data.n_total):
            raise ParameterError("resume data n must be consistent with n_total.")
        if int(data.n_total) < int(self.n_init):
            raise ParameterError("resume data must include at least n_init samples.")

    def _restore_resume_state(self, data):
        self._restore_resume_rng_state(data)
        self.true_measure.discrete_distrib = self.discrete_distrib
        self.integrand.discrete_distrib = self.discrete_distrib
        self.integrand.true_measure.discrete_distrib = self.discrete_distrib

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
            data = Data(parameters=["solution", "comb_bound_low", "comb_bound_high", "comb_bound_diff", "comb_flags", "n_total", "n", "time_integrate"])
            data.flags_indv = np.tile(False, self.integrand.d_indv)
            data.compute_flags = np.tile(True, self.integrand.d_indv)
            data.n = np.tile(self.n_init, self.integrand.d_indv)
            data.n_min = 0
            data.n_max = self.n_init
            data.solution_indv = np.tile(np.nan, self.integrand.d_indv)
            data.xfull = np.empty((0, self.integrand.d))
            data.yfull = np.empty(self.integrand.d_indv + (0,))
        first_resume_iter = resume is not None
        while True:
            if not first_resume_iter:
                xnext = self.discrete_distrib(n=data.n_max - data.n_min)
                data.xfull = np.concatenate([data.xfull, xnext], 0)
                ynext = self.integrand.f(xnext, compute_flags=data.compute_flags)
                ynext[~data.compute_flags] = np.nan
                data.yfull = np.concatenate([data.yfull, ynext], -1)
                data.n[data.compute_flags] = data.n_max
            yfullfinite = np.isfinite(data.yfull)
            data.solution_indv = data.yfull.mean(-1, where=yfullfinite)
            data.sigmahat = data.yfull.std(-1, ddof=1, where=yfullfinite)
            data.ci_half_width = (
                self.z_star * self.inflate * data.sigmahat / np.sqrt(data.n)
            )
            data.indv_bound_low = data.solution_indv - data.ci_half_width
            data.indv_bound_high = data.solution_indv + data.ci_half_width
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
            data.solution[fidxs] = (
                1
                / 2
                * (
                    slow
                    + shigh
                    + self.error_fun(slow, abs_tols, rel_tols)
                    - self.error_fun(shigh, abs_tols, rel_tols)
                )
            )
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
                Note that error tolerances may not be satisfied. """ % (
                    int(data.n_total),
                    int(data.n_total),
                    int(self.n_limit),
                )
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

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rmse_tol is None, "rmse_tol not supported by this stopping criterion."
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb, self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb, self.rel_tol)
