from .abstract_cub_mlmc import AbstractCubMLMC
import copy
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractIIDDiscreteDistribution,
)
from ..integrand import FinancialOption
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
import numpy as np
from scipy.stats import norm
from time import time
import warnings


class CubMLMCCont(AbstractCubMLMC):
    _RESUME_REQUIRED_FIELDS = (
        "levels", "n_level", "sum_level", "diff_n_level", "cost_level", "level_integrands"
    )

    r"""
    Multilevel IID Monte Carlo stopping criterion with continuation.

    Examples:
        >>> fo = FinancialOption(IIDStdUniform(seed=7))
        >>> sc = CubMLMCCont(fo,abs_tol=1.5e-2)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.771
            n_total         2291120
            levels          3
            n_level         [1094715  222428   79666     912     256]
            mean_level      [1.71  0.048 0.012]
            var_level       [21.826  1.768  0.453]
            cost_per_sample [2. 4. 8.]
            alpha           1.970
            beta            1.965
            gamma           1.000
            time_integrate  ...
        CubMLMCCont (AbstractStoppingCriterion)
            rmse_tol        0.006
            n_init          2^(8)
            levels_min      2^(1)
            levels_max      10
            n_tols          10
            inflate         1.668
            theta_init      2^(-1)
            theta           0.010
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
        IIDStdUniform (AbstractIIDDiscreteDistribution)
            d               1
            replications    1
            entropy         7

    **References:**

    1. [https://github.com/PieterjanRobbe/MultilevelEstimators.jl](https://github.com/PieterjanRobbe/MultilevelEstimators.jl).
    """

    def __init__(
        self,
        integrand,
        abs_tol=0.05,
        rmse_tol=None,
        n_init: int = 256,
        n_limit=1e10,
        inflate: float = 100 ** (1 / 9),
        alpha=0.01,
        levels_min: int = 2,
        levels_max: int = 10,
        n_tols: int = 10,
        theta_init: float = 0.5,
    ) -> None:
        r"""Initialize a CubMLMCCont stopping criterion.

        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rmse_tol (np.ndarray): Root mean squared error tolerance. If
                supplied, then absolute tolerance and alpha are ignored in
                favor of the rmse tolerance.
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
            "levels_min",
            "levels_max",
            "n_tols",
            "inflate",
            "theta_init",
            "theta",
        ]
        if levels_min < 2:
            raise ParameterError("needs levels_min >= 2")
        if levels_max < levels_min:
            raise ParameterError("needs levels_max >= levels_min")
        if n_init <= 0:
            raise ParameterError("needs n_init > 0")
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
        self.inflate = inflate
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.alpha0 = -1
        self.beta0 = -1
        self.gamma0 = -1
        self._active_trace = None
        self.alpha = alpha
        self.inflate = inflate
        if not (self.inflate >= 1):
            raise AssertionError
        if not (0 < self.alpha < 1):
            raise AssertionError
        super(CubMLMCCont, self).__init__(
            allowed_distribs=[AbstractIIDDiscreteDistribution],
            allow_vectorized_integrals=False,
        )

    def _validate_resume(self, data):
        self._validate_resume_data(data, required_fields=self._RESUME_REQUIRED_FIELDS)
        self._validate_level_diffs(data)

    def _restore_resume_state(self, data):
        # Undo the final data.levels += 1 stored in the returned data.
        data.levels -= 1

    def _can_replay_resume_exactly(self, data):
        checkpoint_tol = self._checkpoint_rmse_tol(data)
        if checkpoint_tol is None or not (self.target_rmse_tol < checkpoint_tol):
            return False
        return hasattr(data, "level_diffs") and len(data.level_diffs) == len(data.n_level)

    def integrate(self, resume=None) -> tuple:
        """Run (or continue) the continuation-MLMC integration.

        Args:
            resume (Data): Checkpoint returned by a previous ``integrate()``
                call.  The new tolerance may be tighter *or* looser than the
                one used when the checkpoint was created. With a tighter
                tolerance the algorithm picks up the tolerance ladder from
                ``max(checkpoint_rmse_tol, target_rmse_tol)`` and continues
                down to ``target_rmse_tol``.  With a looser tolerance the first
                step immediately converges on the existing samples and no
                additional ladder steps are needed.

        Returns:
            tuple: ``(solution, data)``.
        """
        self._active_t_start = t_start = time()
        self._active_trace = trace = self._make_trace_logger()
        self._active_resume_provenance = resume_provenance = self._capture_resume_provenance(resume)
        try:
            data = self._prepare_resume_data(resume, self._validate_resume, self._restore_resume_state)
            replay_snapshots = None
            replay_iter_count = None
            if self._can_replay_resume_exactly(data):
                replay_data, replay_snapshots, replay_iter_count = self._replay_resume_exactly(
                    data, t_start=t_start, resume_provenance=resume_provenance
                )
                if replay_data is not None:
                    data = replay_data
            if resume is not None:
                checkpoint = self._prepare_resume_data(
                    resume, self._validate_resume, self._restore_resume_state
                )
                step_tol = max(
                    getattr(checkpoint, "rmse_tol", self.target_rmse_tol),
                    self.target_rmse_tol,
                )
                checkpoint.rmse_tol = step_tol
                if replay_iter_count is not None:
                    checkpoint._iter_count = replay_iter_count
                self._set_elapsed_time(checkpoint, 0.0, resume_provenance=resume_provenance)
                trace.resume(checkpoint, step_value=int(checkpoint.levels + 1))
            if replay_snapshots is not None:
                for snapshot in replay_snapshots:
                    trace.iteration(snapshot, step_value=int(snapshot.levels + 1))
                if replay_snapshots and trace.enabled:
                    data._iter_count = trace.iter_count
            elif data is not None:
                step_tol = max(getattr(data, 'rmse_tol', self.target_rmse_tol), self.target_rmse_tol)
                data.rmse_tol = step_tol
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

    @staticmethod
    def _update_trace_solution(data):
        valid = data.n_level[: data.levels + 1] > 0
        if np.any(valid):
            data.solution = (
                data.sum_level[0, : data.levels + 1][valid]
                / data.n_level[: data.levels + 1][valid]
            ).sum()

    def _replay_resume_exactly(self, checkpoint, t_start=None, resume_provenance=None):
        """Ensure iteration number in `replay_iter_count` same in LOOSE-last
        and RESUMED-first iterations, by simply saving `level_rep_sums` and
        `level_n_increments`.
        """
        shadow_trace = self._active_trace = None
        try:
            shadow = self._construct_data()
            shadow.level_integrands = list(checkpoint.level_integrands)
            shadow.cached_level_diffs = [
                np.asarray(level_diffs, dtype=float) for level_diffs in checkpoint.level_diffs
            ]
            shadow.cached_level_positions = np.zeros(
                len(shadow.cached_level_diffs), dtype=int
            )
            snapshots = []
            for t in range(self.n_tols):
                step_tol = self.inflate ** (self.n_tols - t - 1) * self.target_rmse_tol
                self._integrate(
                    shadow,
                    step_tol=step_tol,
                    record_snapshots=True,
                    snapshots=snapshots,
                    update_data_fn=self._update_replay_data,
                    t_start=t_start,
                    resume_provenance=resume_provenance,
                )
        finally:
            self._active_trace = shadow_trace
        target_counts = np.asarray(checkpoint.n_level[: checkpoint.levels + 1], dtype=int)
        absorb_index = None
        for i, snapshot in enumerate(snapshots):
            if len(snapshot.n_level) < len(target_counts):
                continue
            if np.all(snapshot.n_level[: len(target_counts)] >= target_counts):
                absorb_index = i
                break
        if absorb_index is None:
            return None, None, None
        same_state = (
            np.array_equal(
                np.asarray(snapshots[absorb_index].n_level[: len(target_counts)]),
                target_counts,
            )
            and np.array_equal(
                np.asarray(snapshots[absorb_index].sum_level[:, : len(target_counts)]),
                np.asarray(checkpoint.sum_level[:, : len(target_counts)]),
            )
        )
        replay_iter_count = absorb_index + 1 if same_state else absorb_index
        for attr in ("cached_level_diffs", "cached_level_positions"):
            if hasattr(shadow, attr):
                delattr(shadow, attr)
        return shadow, snapshots[absorb_index:], replay_iter_count

    def _integrate(
        self,
        data,
        skip_level_reset=False,
        trace=None,
        step_tol=None,
        record_snapshots=False,
        snapshots=None,
        update_data_fn=None,
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
        update_data_fn = self._update_data if update_data_fn is None else update_data_fn
        self.theta = self.theta_init
        if not skip_level_reset:
            data.levels = int(self.levels_min)
        update_data_fn(data)  # Take warm-up samples if none have been taken so far
        warmup_drew = data.diff_n_level.sum() > 0
        self._update_trace_solution(data)
        if trace is not None and warmup_drew:
            data.rmse_tol = step_tol
            if t_start is not None:
                self._set_elapsed_time(
                    data, time() - t_start, resume_provenance=resume_provenance
                )
            trace.iteration(data, step_value=int(data.levels + 1))
        if record_snapshots and warmup_drew:
            data.rmse_tol = step_tol
            if t_start is not None:
                self._set_elapsed_time(
                    data, time() - t_start, resume_provenance=resume_provenance
                )
            snapshots.append(copy.deepcopy(data))

        converged = False
        while not converged:

            # Check if we already have samples at the finest level
            if not data.n_level[data.levels] > 0:
                # This takes n_init warm-up samples at the finest level
                data.diff_n_level = np.hstack((data.diff_n_level, self.n_init))
                update_data_fn(data)
                self._update_trace_solution(data)
                if trace is not None:
                    data.rmse_tol = step_tol
                    if t_start is not None:
                        self._set_elapsed_time(
                            data, time() - t_start, resume_provenance=resume_provenance
                        )
                    trace.iteration(data, step_value=int(data.levels + 1))
                if record_snapshots:
                    data.rmse_tol = step_tol
                    if t_start is not None:
                        self._set_elapsed_time(
                            data, time() - t_start, resume_provenance=resume_provenance
                        )
                    snapshots.append(copy.deepcopy(data))
                # Alternatively, evaluate optimal number of samples and take between 2 and n_init samples
                # data.diff_n_level = self._get_next_samples(data)
                # data.diff_n_level[:data.levels] = 0
                # data.diff_n_level[data.levels] = max(3, min(self.n_init, data.diff_n_level[data.levels]))

            # Update splitting parameter
            self._update_theta(data, step_tol)

            # Set optimal number of additional samples
            n_samples = self._get_next_samples(data)
            data.diff_n_level = np.maximum(
                0, n_samples - data.n_level[: data.levels + 1]
            )

            # Check if over sample budget
            if (data.n_total + data.diff_n_level.sum()) > self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples, which would exceed n_limit = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" % (
                    int(data.n_total),
                    int(data.diff_n_level.sum()),
                    int(self.n_limit),
                )
                warnings.warn(warning_s, MaxSamplesWarning)
                break

            # Take additional samples
            update_data_fn(data)
            self._update_trace_solution(data)
            if trace is not None:
                data.rmse_tol = step_tol
                if t_start is not None:
                    self._set_elapsed_time(
                        data, time() - t_start, resume_provenance=resume_provenance
                    )
                trace.iteration(data, step_value=int(data.levels + 1))
            if record_snapshots:
                data.rmse_tol = step_tol
                if t_start is not None:
                    self._set_elapsed_time(
                        data, time() - t_start, resume_provenance=resume_provenance
                    )
                snapshots.append(copy.deepcopy(data))

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

        data.diff_n_level.fill(0)
        data.solution = (
            data.sum_level[0, : data.levels + 1] / data.n_level[: data.levels + 1]
        ).sum()
        data.levels += 1

    def _update_theta(self, data, step_tol):
        # Update error splitting parameter
        self.theta = max(
            0.01,
            min(0.5, (self._bias(data, len(data.n_level) - 1) / step_tol) ** 2),
        )

    def _rmse(self, data):
        # Returns an estimate for the root mean square error
        return np.sqrt(self._mse(data))

    def _mse(self, data):
        # Returns an estimate for the mean square error
        return (1 - self.theta) * self._varest(data) + self.theta * self._bias(
            data, data.levels
        ) ** 2

    def _varest(self, data):
        # Returns the variance of the estimator
        return (data.var_level / data.n_level[: data.levels + 1]).sum()

    def _bias(self, data, level):
        # Returns an estimate for the bias
        mean_level = data.sum_level[0, :] / data.n_level
        A = np.ones((2, 2))
        A[:, 0] = range(level - 1, level + 1)
        y = np.log2(abs(mean_level[level - 1 : level + 1]))
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha = max(0.5, -x[0])
        return 2 ** (x[1] + (level + 1) * x[0]) / (2**alpha - 1)
