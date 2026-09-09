from typing import Union
from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data

from ..util import MaxSamplesWarning, ParameterError, ParameterWarning
import numpy as np
from time import time
import warnings
from scipy.optimize import fminbound as fminbnd
from scipy.optimize import fmin
from scipy.stats import norm as gaussnorm
from scipy.stats import t as tnorm


class AbstractCubBayesLDG(AbstractStoppingCriterion):
    """Abstract base class for guaranteed Bayesian low-discrepancy QMC stopping criteria.

    Implements the fast-transform Bayesian cubature error bound shared by
    concrete lattice/digital-net Bayesian stopping criteria: doubling sample
    counts each iteration, maintaining the running transform coefficients
    (`_ytildefull`), and fitting the kernel hyperparameter
    (`objective_function`, `_stopping_criterion`) to derive a
    credible-interval error bound.
    """

    _RESUME_REQUIRED_FIELDS = (
        "solution", "comb_bound_low", "comb_bound_high", "comb_bound_diff", "comb_flags", "n", "n_max", "xfull", "yfull"
    )
    _RESUME_STATE_FIELDS = ("_ytildefull",)

    def __init__(
        self,
        integrand,
        ft,
        omega,
        ptransform,
        allowed_distribs,
        kernel,
        abs_tol,
        rel_tol,
        n_init,
        n_limit,
        alpha,
        error_fun,
        errbd_type,
    ) -> None:
        self.parameters = ["abs_tol", "rel_tol", "n_init", "n_limit", "order"]
        # Input Checks
        if np.log2(n_init) % 1 != 0:
            warnings.warn(
                "n_init must be a power of two. Using n_init = 2**5", ParameterWarning
            )
            n_init = 2**8
        if np.log2(n_limit) % 1 != 0:
            warnings.warn(
                "n_init must be a power of two. Using n_limit = 2**30", ParameterWarning
            )
            n_limit = 2**22
        # Set Attributes
        self.n_init = int(n_init)
        self.n_limit = int(n_limit)
        if not (isinstance(error_fun, str) or callable(error_fun)):
            raise AssertionError
        # _error_fun_key stores a simple, serializable string and ensures correct state saving
        # in __getstate__(), bypassing serialization of complex lambda functions, which often fails.
        self.error_fun, self._error_fun_key = self._resolve_error_fun(error_fun)
        self.alpha = alpha
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(AbstractCubBayesLDG, self).__init__(
            allowed_distribs=allowed_distribs, allow_vectorized_integrals=True
        )
        if not (
            self.integrand.discrete_distrib.no_replications == True
        ):
            raise AssertionError("Require the discrete distribution has replications=None")
        if not (
            self.integrand.discrete_distrib.randomize != "FALSE"
        ):
            raise AssertionError("Require discrete distribution is randomized")
        self.alphas_indv, _ = self._compute_indv_alphas(
            np.full(self.integrand.d_comb, self.alpha)
        )
        self.set_tolerance(abs_tol, rel_tol)
        self.stop_at_tol = (
            True  # automatic mode: stop after meeting the error tolerance
        )
        self.arb_mean = True  # by default use zero mean algorithm
        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.debug_enable = False  # enable debug prints
        self.use_gradient = False  # If true uses gradient descent in parameter search
        self.one_theta = True  # If true use common shape parameter for all dimensions, else allow shape parameter vary across dimensions
        self.errbd_type = errbd_type.upper()
        if not (self.errbd_type in ["MLE", "GCV", "FULL"]):
            raise AssertionError
        self.kernel = kernel
        self.debugEnable = True
        self.ft = ft
        self.omega = omega
        self.ptransform = ptransform  # periodization transform
        if self.errbd_type == "FULL":
            self.uncert = -tnorm.ppf(alpha / 2, self.n_init - 1)
        else:
            self.uncert = -gaussnorm.ppf(alpha / 2)

    def _stopping_criterion(self, xpts, ftilde, m):
        ftilde = ftilde.squeeze()
        n = 2**m
        lna_range = [
            -5,
            0,
        ]  # reduced from [-5, 5], to avoid kernel values getting too big causing error

        # search for optimal shape parameter
        if self.one_theta == True:
            lna_MLE = fminbnd(
                lambda lna: self.objective_function(np.exp(lna), xpts, ftilde)[0],
                x1=lna_range[0],
                x2=lna_range[1],
                xtol=1e-2,
                disp=0,
            )

            aMLE = np.exp(lna_MLE)
            _, vec_lambda, vec_lambda_ring, RKHS_norm = self.objective_function(
                aMLE, xpts, ftilde
            )
        else:
            if self.use_gradient == True:
                warnings.warn("Not implemented !")
                lna_MLE = 0
            else:
                # Nelder-Mead Simplex algorithm
                theta0 = np.ones((xpts.shape[1], 1)) * (0.05)
                theta0 = np.ones((1, xpts.shape[1])) * (0.05)
                lna_MLE = fmin(
                    lambda lna: self.objective_function(np.exp(lna), xpts, ftilde)[0],
                    theta0,
                    xtol=1e-2,
                    disp=False,
                )
            aMLE = np.exp(lna_MLE)
            # print(n, aMLE)
            _, vec_lambda, vec_lambda_ring, RKHS_norm = self.objective_function(
                aMLE, xpts, ftilde
            )

        # Check error criterion
        # compute DSC
        if self.errbd_type == "FULL":
            # full Bayes
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / n)
            else:
                DSC = abs((vec_lambda[0] / n) - 1)

            # 1-alpha two sided confidence interval
            err_bd = self.uncert * np.sqrt(DSC * RKHS_norm / (n - 1))
        elif self.errbd_type == "GCV":
            # GCV based stopping criterion
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / (n + vec_lambda_ring[0]))
            else:
                DSC = abs(1 - (n / vec_lambda[0]))

            temp = vec_lambda
            temp[0] = n + vec_lambda_ring[0]
            mC_inv_trace = np.sum(1.0 / temp(temp != 0))
            err_bd = self.uncert * np.sqrt(DSC * RKHS_norm / mC_inv_trace)
        else:
            # empirical Bayes
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / (n + vec_lambda_ring[0]))
            else:
                DSC = abs(1 - (n / vec_lambda[0]))
            err_bd = self.uncert * np.sqrt(DSC * RKHS_norm / n)

        if self.arb_mean:  # zero mean case
            muhat = ftilde[0] / n
        else:  # non zero mean case
            muhat = ftilde[0] / vec_lambda[0]

        self.error_bound = err_bd
        muhat = np.abs(muhat)
        return muhat, err_bd

    def __getstate__(self):
        state = self.__dict__.copy()
        # error_fun is a local lambda when constructed from a string keyword.
        # Replace with the canonical string form so it can be reconstructed.
        if self._error_fun_key is not None:
            state['error_fun'] = self._error_fun_key
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild error_fun from its string key if present.
        if isinstance(self.error_fun, str):
            self.error_fun, _ = self._resolve_error_fun(self.error_fun)

    def objective_function(self, theta: float, xun: np.ndarray, ftilde: np.ndarray) -> float:
        """Compute the Bayesian cubature loss used to fit the kernel parameter theta.

        Evaluates either the negative log marginal likelihood (MLE) or the
        generalized cross validation (GCV) loss, per `self.errbd_type`, along
        with the kernel eigenvalues and RKHS norm needed by the error bound.

        Args:
            theta (float): Kernel hyperparameter to evaluate the loss at.
            xun (np.ndarray): Unique ordered node locations.
            ftilde (np.ndarray): Fast-transformed function values at `xun`.

        Returns:
            float: Loss value (MLE or GCV, per `self.errbd_type`).
            np.ndarray: Kernel eigenvalues `vec_lambda`.
            np.ndarray: Ring eigenvalues `vec_lambda_ring`, used by the error bound.
            float: RKHS norm estimate of the fitted function.
        """
        n = len(ftilde)
        fudge = 100 * np.finfo(float).eps
        # if type(theta) != np.ndarray:
        #     theta = np.ones((1, xun.shape[1])) * theta
        [vec_lambda, vec_lambda_ring, lambda_factor] = self.kernel(
            xun,
            self.order,
            theta,
            self.avoid_cancel_error,
            self.kernType,
            self.debug_enable,
        )
        vec_lambda = abs(vec_lambda)
        # compute RKHS_norm
        temp = abs(ftilde[vec_lambda > fudge] ** 2) / (vec_lambda[vec_lambda > fudge])

        # compute loss
        if self.errbd_type == "GCV":
            # GCV
            temp_gcv = (
                abs(ftilde[vec_lambda > fudge] / (vec_lambda[vec_lambda > fudge])) ** 2
            )
            loss1 = 2 * np.log(sum(1.0 / vec_lambda[vec_lambda > fudge]))
            loss2 = np.log(sum(temp_gcv[1:]))
            # ignore all zero eigenvalues
            loss = loss2 - loss1

            if self.arb_mean:
                RKHS_norm = (1 / lambda_factor) * sum(temp_gcv[1:]) / n
            else:
                RKHS_norm = (1 / lambda_factor) * sum(temp_gcv) / n
        else:
            # default: MLE
            if self.arb_mean:
                RKHS_norm = (1 / lambda_factor) * sum(temp[1:]) / n
                temp_1 = (1 / lambda_factor) * sum(temp[1:])
            else:
                RKHS_norm = (1 / lambda_factor) * sum(temp) / n
                temp_1 = (1 / lambda_factor) * sum(temp)

            # ignore all zero eigenvalues
            loss1 = sum(np.log(abs(lambda_factor * vec_lambda[vec_lambda > fudge])))
            if temp_1 != 0:
                loss2 = n * np.log(temp_1)
            else:
                loss2 = n * np.log(temp_1 + np.finfo(float).eps)
            loss = loss1 + loss2

        if self.debug_enable:
            self.alert_msg(loss1, "Inf", "Imag")
            self.alert_msg(RKHS_norm, "Imag")
            self.alert_msg(loss2, "Inf", "Imag")
            self.alert_msg(loss, "Inf", "Imag", "Nan")
            self.alert_msg(vec_lambda, "Imag")

        vec_lambda, vec_lambda_ring = (
            lambda_factor * vec_lambda,
            lambda_factor * vec_lambda_ring,
        )
        return loss, vec_lambda, vec_lambda_ring, RKHS_norm

    @staticmethod
    def kernel_t(aconst: Union[float, np.ndarray], Bern: np.ndarray) -> np.ndarray:
        r"""Compute the modified kernel ``Km1 = K - 1`` from Bernoulli polynomial values.

        Working with ``Km1`` rather than ``K`` directly avoids cancellation
        error when later computing $1 - n/\lambda_1$.

        Args:
            aconst (Union[float, np.ndarray]): Kernel parameter theta, scalar
                or per-dimension array.
            Bern (np.ndarray): Bernoulli polynomial values, shape ``(n, d)``.

        Returns:
            np.ndarray: ``Km1``, the kernel minus one.
            np.ndarray: ``K``, the full kernel (``1 + Km1``).
        """
        d = np.size(Bern, 1)
        if type(aconst) != np.ndarray:
            theta = np.ones((d, 1)) * aconst
        else:
            theta = aconst  # theta varies per dimension

        Kjm1 = theta[0] * Bern[:, 0]  # Kernel at j-dim minus One
        Kj = 1 + Kjm1  # Kernel at j-dim

        for j in range(1, d):
            Kjm1_prev = Kjm1
            Kj_prev = Kj  # save the Kernel at the prev dim

            Kjm1 = theta[j] * Bern[:, j] * Kj_prev + Kjm1_prev
            Kj = 1 + Kjm1

        Km1 = Kjm1
        K = Kj
        return [Km1, K]

    @staticmethod
    def alert_msg(*args: tuple):
        """Print a debug message if a variable contains NaN, Inf, or complex values.

        Args:
            *args (tuple): The variable to check, followed by one or more of
                ``"Nan"``, ``"Inf"``, ``"Imag"`` naming which conditions to
                report. Example: `alert_msg(x, "Inf", "Imag")` prints if `x`
                contains infinite or imaginary values.
        """
        varargin = args
        nargin = len(varargin)
        if nargin > 1:
            i_start = 0
            var_tocheck = varargin[i_start]
            i_start = i_start + 1
            inpvarname = "variable"

            while i_start < nargin:
                var_type = varargin[i_start]
                i_start = i_start + 1

                if var_type == "Nan":
                    if np.any(np.isnan(var_tocheck)):
                        print("%s has NaN values" % inpvarname)
                elif var_type == "Inf":
                    if np.any(np.isinf(var_tocheck)):
                        print("%s has Inf values" % inpvarname)
                elif var_type == "Imag":
                    if not np.all(np.isreal(var_tocheck)):
                        print("%s has complex values" % inpvarname)
                else:
                    print("unknown type check requested !")

    def integrate(self, resume: Union[None, Data] = None) -> tuple:
        """Determine the samples needed to satisfy the target tolerance.

        Doubles the sample count each iteration, updates the running fast
        transform (`_ytildefull`), and for each not-yet-converged output
        calls `_stopping_criterion` to fit the Bayesian kernel hyperparameter
        and derive a credible-interval bound on the integral. Stops once
        every combined output is within tolerance or `self.n_limit` would be
        exceeded.

        Args:
            resume (Union[None, Data]): Existing integration state to resume from, if
                supported. Defaults to None.

        Returns:
            tuple: Approximation to the integral with shape ``integrand.d_comb``
                and the corresponding data object.
        """
        t_start = time()
        resume_provenance = self._capture_resume_provenance(resume)
        first_resume_iter = False
        trace = self._make_trace_logger()
        data = self._prepare_resume_data(
            resume, self._validate_resume, self._restore_resume_state
        )
        if data is not None:
            data.flags_indv = np.tile(False, self.integrand.d_indv)
            data.compute_flags = np.tile(True, self.integrand.d_indv)
            data.n_min = int(data.n_total)
            ytildefull = data._ytildefull
            first_resume_iter = True
            self._set_elapsed_time(data, 0.0, resume_provenance=resume_provenance)
            trace.resume(data, step_value=int(np.log2(max(1, int(data.n_total)))))
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
            data.bounds_half_width = np.tile(np.inf, self.integrand.d_indv)
            data.muhat = np.tile(np.nan, self.integrand.d_indv)
        while True:
            m = int(np.log2(data.n_max))
            if not first_resume_iter:
                xnext = self.discrete_distrib(n_min=data.n_min, n_max=data.n_max)
                data.xfull = np.concatenate([data.xfull, xnext], 0)
                ynext = self.integrand.f(
                    xnext,
                    periodization_transform=self.ptransform,
                    compute_flags=data.compute_flags,
                )
                ynext[~data.compute_flags] = np.nan
                data.yfull = np.concatenate([data.yfull, ynext], -1)
            if not first_resume_iter and data.n_min == 0:  # first fresh iteration
                ytildefull = self.ft(ynext) / np.sqrt(2**m)
            elif not first_resume_iter:  # any iteration after the first
                mnext = int(m - 1)
                ytildeomega = (
                    self.omega(mnext)
                    * self.ft(ynext[data.compute_flags])
                    / np.sqrt(2**mnext)
                )
                ytildefull_next = np.nan * np.ones_like(ytildefull)
                ytildefull_next[data.compute_flags] = (
                    ytildefull[data.compute_flags] - ytildeomega
                ) / 2
                ytildefull[data.compute_flags] = (
                    ytildefull[data.compute_flags] + ytildeomega
                ) / 2
                ytildefull = np.concatenate([ytildefull, ytildefull_next], axis=-1)
            for j in np.ndindex(self.integrand.d_indv):
                if not data.compute_flags[j]:
                    continue
                data.muhat[j], data.bounds_half_width[j] = self._stopping_criterion(
                    data.xfull, 2**m * ytildefull[j], m
                )
            data.indv_bound_low = data.muhat - data.bounds_half_width
            data.indv_bound_high = data.muhat + data.bounds_half_width
            data.n[data.compute_flags] = data.n_max
            data.n_total = data.n_max
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
            # Save transform state so this computation can be resumed later.
            data._ytildefull = ytildefull
            self._set_elapsed_time(data, time() - t_start, resume_provenance=resume_provenance)
            trace.iteration(data, step_value=m)
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

    def _validate_resume(self, data):
        self._validate_resume_with_state(
            data,
            required_fields=self._RESUME_REQUIRED_FIELDS,
            state_fields=self._RESUME_STATE_FIELDS,
        )
        n_total = int(data.n_total)
        output_shape = self.integrand.d_indv + (n_total,)
        self._validate_resume_shape("xfull", data.xfull, (n_total, self.integrand.d))
        self._validate_resume_shape("yfull", data.yfull, output_shape)
        self._validate_resume_shape("_ytildefull", data._ytildefull, output_shape)
        self._validate_resume_shape("n", data.n, self.integrand.d_indv)
        if int(np.max(np.asarray(data.n))) != n_total:
            raise ParameterError("resume data n must be consistent with n_total.")
        if int(data.n_max) != n_total:
            raise ParameterError("resume data n_total must match n_max.")
        if not self._is_power_of_two(n_total):
            raise ParameterError("resume data n_total must be a power of 2.")

    def set_tolerance(self, abs_tol: Union[None, float] = None, rel_tol: Union[None, float] = None, rmse_tol: Union[None, float] = None):
        """Update the stopping criterion's target tolerance.

        Args:
            abs_tol (Union[None, float]): Absolute error tolerance, broadcast to
                `self.abs_tols` with shape `integrand.d_comb`.
            rel_tol (Union[None, float]): Relative error tolerance, broadcast to
                `self.rel_tols` with shape `integrand.d_comb`.
            rmse_tol (Union[None, float]): Unsupported; must be `None`.

        Raises:
            AssertionError: If `rmse_tol` is supplied.
        """
        if not (rmse_tol is None):
            raise AssertionError("rmse_tol not supported by this stopping criterion.")
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb, self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb, self.rel_tol)
