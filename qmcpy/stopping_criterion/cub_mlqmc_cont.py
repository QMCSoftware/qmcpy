from .abstract_cub_mlqmc import AbstractCubMLQMC
from ..util.data import Data
import copy
from ..discrete_distribution import DigitalNetB2, Lattice, Halton
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractLDDiscreteDistribution,
)
from ..integrand import FinancialOption
from ..util import MaxSamplesWarning, MaxLevelsWarning, ParameterError
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMLQMCCont(AbstractCubMLQMC):
    _RESUME_REQUIRED_FIELDS = (
        "levels", "n_level", "eval_level", "mean_level_reps", "mean_level",
        "var_level", "cost_level", "var_cost_ratio_level", "bias_estimate", "level_integrands"
    )

    """
    Multilevel Quasi-Monte Carlo stopping criterion with continuation.

    Examples:
        >>> fo = FinancialOption(DigitalNetB2(seed=7,replications=32))
        >>> sc = CubMLQMCCont(fo,abs_tol=1e-3)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.784
            n_total         4718592
            levels          2^(2)
            n_level         [65536 32768 32768 16384]
            mean_level      [1.718 0.051 0.012 0.003]
            var_level       [1.169e-08 2.569e-08 1.850e-08 5.209e-08]
            bias_estimate   2.78e-04
            time_integrate  ...
        CubMLQMCCont (AbstractStoppingCriterion)
            rmse_tol        3.88e-04
            n_init          2^(8)
            n_limit         10000000000
            replications    2^(5)
            levels_min      2^(1)
            levels_max      10
            n_tols          10
            inflate         1.668
            theta_init      2^(-1)
            theta           2^(-3)
        FinancialOption (AbstractIntegrand)
            option          ASIAN
            call_put        CALL
            volatility      2^(-1)
            start_price     30
            strike_price    35
            interest_rate   0
            t_final         1
            asian_mean      ARITHMETIC
        GeometricBrownianMotion (AbstractTrueMeasure)
            time_vec        1
            drift           0
            diffusion       2^(-2)
            mean_gbm        30
            covariance_gbm  255.623
            decomp_type     PCA
        DigitalNetB2 (AbstractLDDiscreteDistribution)
            d               1
            replications    2^(5)
            randomize       LMS DS
            gen_mats_source joe_kuo.6.21201.txt
            order           RADICAL INVERSE
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7

    **References:**

    1.  [https://github.com/PieterjanRobbe/MultilevelEstimators.jl](https://github.com/PieterjanRobbe/MultilevelEstimators.jl).
    """

    def __init__(
        self,
        integrand,
        abs_tol=0.05,
        rmse_tol=None,
        n_init=256,
        n_limit=1e10,
        inflate=100 ** (1 / 9),
        alpha=0.01,
        levels_min=2,
        levels_max=10,
        n_tols=10,
        theta_init=0.5,
    ):
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rmse_tol (np.ndarray): Root mean squared error tolerance.
                If supplied, then absolute tolerance and alpha are ignored in favor of the rmse tolerance.
            n_init (int): Initial number of samples.
            n_limit (int): Maximum number of samples.
            inflate (float): Coarser tolerance multiplication factor $\geq 1$.
            alpha (np.ndarray): Uncertainty level in $(0,1)$.
            levels_min (int): Minimum level of refinement $\geq 2$.
            levels_max (int): Maximum level of refinement $\geq$ `levels_min`.
            n_tols (int): Number of coarser tolerances to run.
            theta_init (float): Initial error splitting constant.
        """
        self.parameters = [
            "rmse_tol",
            "n_init",
            "n_limit",
            "replications",
            "levels_min",
            "levels_max",
            "n_tols",
            "inflate",
            "theta_init",
            "theta",
        ]
        # initialization
        if rmse_tol:
            self.target_rmse_tol = float(rmse_tol)
        else:  # use absolute tolerance
            self.target_rmse_tol = float(abs_tol) / norm.ppf(1 - alpha / 2)
        self.rmse_tol = self.target_rmse_tol  # user-facing attribute; never mutated after __init__
        self.n_init = n_init
        self.n_limit = n_limit
        self.levels_min = levels_min
        self.levels_max = levels_max
        self.theta_init = theta_init
        self.theta = theta_init
        self.n_tols = n_tols
        self._active_trace = None
        self.alpha = alpha
        self.inflate = inflate
        assert self.inflate >= 1
        assert 0 < self.alpha < 1
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMLQMCCont, self).__init__(
            allowed_distribs=[AbstractLDDiscreteDistribution],
            allow_vectorized_integrals=False,
        )
        self.replications = self.discrete_distrib.replications
        assert self.replications >= 4, "require at least 4 replications"

    def _validate_resume(self, data):
        self._validate_resume_data(data, required_fields=self._RESUME_REQUIRED_FIELDS)
        if np.shape(data.mean_level_reps) != (data.levels, int(self.replications)):
            raise ParameterError("resume data mean_level_reps shape %s is incompatible with levels=%d, replications=%d." % (np.shape(data.mean_level_reps), data.levels, int(self.replications)))
        try:
            self._validate_level_replay_cache(data)
        except ValueError as exc:
            raise ParameterError(str(exc)) from exc

    def _restore_resume_state(self, data):
        # No data.levels adjustment needed for MLQMC (no final += 1 in this variant).
        pass

    def _can_replay_resume_exactly(self, data):
        checkpoint_tol = self._checkpoint_rmse_tol(data)
        if checkpoint_tol is None or not (self.target_rmse_tol < checkpoint_tol):
            return False
        return hasattr(data, "level_rep_sums") and hasattr(data, "level_n_increments")

    def integrate(self, resume=None) -> tuple:
        """Run (or continue) the continuation-MLQMC integration.

        Args:
            resume (Data, optional): Checkpoint returned by a previous
                ``integrate()`` call.  The new tolerance may be tighter *or*
                looser than the one used when the checkpoint was created.
                With a tighter tolerance the algorithm picks up the tolerance
                ladder from ``max(checkpoint_rmse_tol, target_rmse_tol)`` and
                continues down to ``target_rmse_tol``.  With a looser tolerance
                the first step immediately converges on the existing samples
                and no additional ladder steps are needed.

        Returns:
            tuple: ``(solution, data)``.
        """
        self._active_t_start = t_start = time()
        self._active_trace = trace = self._make_trace_logger()
        self._active_resume_provenance = resume_provenance = self._capture_resume_provenance(resume)
        try:
            data = self._prepare_resume_data(resume, self._validate_resume, self._restore_resume_state)
            if data is not None:
                step_tol = max(getattr(data, 'rmse_tol', self.target_rmse_tol), self.target_rmse_tol)
                data.rmse_tol = step_tol
                self._set_elapsed_time(data, 0.0, resume_provenance=resume_provenance)
                trace.resume(data, step_value=int(data.levels))
                if step_tol <= self.target_rmse_tol:
                    # Same or looser tolerance: check convergence at target_tol
                    self._integrate(
                        data,
                        skip_level_reset=True,
                        step_tol=step_tol,
                    )
                else:
                    # Tighter tolerance: skip to ladder, preserving level structure for first step
                    first = True
                    for t in range(self.n_tols):
                        next_tol = self.inflate ** (self.n_tols - t - 1) * self.target_rmse_tol
                        if next_tol < step_tol:
                            self._integrate(
                                data,
                                skip_level_reset=first,
                                step_tol=next_tol,
                            )
                            first = False
            else:
                data = self._construct_data()
                # Loop over coarser tolerances
                for t in range(self.n_tols):
                    step_tol = self.inflate ** (self.n_tols - t - 1) * self.target_rmse_tol
                    self._integrate(
                        data,
                        step_tol=step_tol,
                    )
            self._finalize_integration_data(
                data, time() - t_start, resume_provenance=resume_provenance
            )
            trace.finalize()
            data.iteration_history = getattr(self, "iteration_history", None)
            data.history_df = getattr(self, "history_df", None)
            return data.solution, data
        finally:
            self._active_trace = None
            self._active_t_start = None
            self._active_resume_provenance = None

    def _construct_data(self):
        data = Data(parameters=["solution", "n_total", "levels", "n_level", "mean_level", "var_level", "bias_estimate"])
        data.levels = int(self.levels_min + 1)
        data.n_level = np.zeros(data.levels, dtype=int)
        data.eval_level = np.ones(data.levels, dtype=bool)
        data.mean_level_reps = np.zeros((data.levels, int(self.replications)))
        data.mean_level = np.tile(0.0, data.levels)
        data.var_level = np.tile(np.inf, data.levels)
        data.cost_level = np.tile(0.0, data.levels)
        data.var_cost_ratio_level = np.tile(np.inf, data.levels)
        data.bias_estimate = np.inf
        data.level_integrands = []
        data.level_rep_sums = [[] for _ in range(data.levels)]
        data.level_n_increments = [[] for _ in range(data.levels)]
        return data

    def _integrate(
        self,
        data,
        skip_level_reset=False,
        trace=None,
        step_tol=None,
        record_snapshots=False,
        snapshots=None,
        update_data_fn=None,
        stop_condition=None,
        t_start=None,
        resume_provenance=None,
    ):
        t_start = getattr(self, "_active_t_start", None) if t_start is None else t_start
        resume_provenance = (
            getattr(self, "_active_resume_provenance", None)
            if resume_provenance is None
            else resume_provenance
        )
        if step_tol is None:
            step_tol = self.rmse_tol
        trace = getattr(self, "_active_trace", None) if trace is None else trace
        update_data_fn = self.update_data if update_data_fn is None else update_data_fn
        snapshots = [] if snapshots is None else snapshots
        # self.theta = self.theta_init
        if not skip_level_reset:
            data.levels = int(self.levels_min + 1)

        converged = False
        while not converged:
            # Ensure that we have samples on the finest level
            update_data_fn(data)
            if t_start is not None:
                self._set_elapsed_time(
                    data, time() - t_start, resume_provenance=resume_provenance
                )
            if trace is not None:
                data.rmse_tol = step_tol
                trace.iteration(data, step_value=int(data.levels))
            if record_snapshots:
                data.rmse_tol = step_tol
                snapshots.append(copy.deepcopy(data))
            if stop_condition is not None and stop_condition(data):
                return True
            self._update_theta(data, step_tol)

            while self._varest(data) > (1 - self.theta) * step_tol**2:
                efficient_level = np.argmax(data.var_cost_ratio_level[: data.levels])
                data.eval_level[efficient_level] = True

                # Check if over sample budget
                total_next_samples = (
                    self.replications * data.eval_level * data.n_level * 2
                ).sum()
                if (data.n_total + total_next_samples) > self.n_limit:
                    warning_s = """
                    Already generated %d samples.
                    Trying to generate %d new samples, which would exceed n_limit = %d.
                    Stopping integration process.
                    Note that error tolerances may no longer be satisfied""" % (
                        int(data.n_total),
                        int(total_next_samples),
                        int(self.n_limit),
                    )
                    warnings.warn(warning_s, MaxSamplesWarning)
                    return

                update_data_fn(data)
                if t_start is not None:
                    self._set_elapsed_time(
                        data, time() - t_start, resume_provenance=resume_provenance
                    )
                if trace is not None:
                    data.rmse_tol = step_tol
                    trace.iteration(data, step_value=int(data.levels))
                if record_snapshots:
                    data.rmse_tol = step_tol
                    snapshots.append(copy.deepcopy(data))
                if stop_condition is not None and stop_condition(data):
                    return True
                self._update_theta(data, step_tol)

            # Check for convergence
            converged = self._rmse(data) < step_tol
            if not converged:
                if data.levels == self.levels_max:
                    warnings.warn(
                        "Failed to achieve weak convergence. levels == levels_max.",
                        MaxLevelsWarning,
                    )
                    converged = True
                else:
                    self._add_level(data)
        return False

    def _update_theta(self, data, step_tol):
        # Update error splitting parameter
        max_levels = len(data.n_level)
        A = np.ones((2, 2))
        A[:, 0] = range(max_levels - 2, max_levels)
        y = np.ones(2)
        y[0] = np.log2(abs(data.mean_level_reps[max_levels - 2].mean()))
        y[1] = np.log2(abs(data.mean_level_reps[max_levels - 1].mean()))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(0.5, -x[0])
        real_bias = 2 ** (x[1] + max_levels * x[0]) / (2**alpha - 1)
        self.theta = max(0.01, min(0.125, (real_bias / step_tol) ** 2))

    def _rmse(self, data):
        # Returns an estimate for the root mean square error
        return np.sqrt(self._mse(data))

    def _mse(self, data):
        # Returns an estimate for the mean square error
        return (1 - self.theta) * self._varest(
            data
        ) + self.theta * data.bias_estimate**2

    def _varest(self, data):
        # Returns the variance of the estimator
        return data.var_level[: data.levels].sum()
