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


class CubMLQMC(AbstractCubMLQMC):
    _RESUME_REQUIRED_FIELDS = (
        "levels", "n_level", "eval_level", "mean_level_reps", "mean_level",
        "var_level", "cost_level", "var_cost_ratio_level", "bias_estimate", "level_integrands"
    )

    """
    Multilevel Quasi-Monte Carlo stopping criterion.

    Examples:
        >>> fo = FinancialOption(DigitalNetB2(seed=7,replications=32))
        >>> sc = CubMLQMC(fo,abs_tol=3e-3)
        >>> solution,data = sc.integrate()
        >>> data
        Data (Data)
            solution        1.784
            n_total         2359296
            levels          2^(2)
            n_level         [32768 16384 16384  8192]
            mean_level      [1.718 0.051 0.012 0.003]
            var_level       [7.119e-08 1.409e-07 9.668e-08 1.852e-07]
            bias_estimate   2.99e-04
            time_integrate  ...
        CubMLQMC (AbstractStoppingCriterion)
            rmse_tol        0.001
            n_init          2^(8)
            n_limit         10000000000
            replications    2^(5)
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

    1.  M.B. Giles and B.J. Waterhouse.
        'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics, de Gruyter, 2009.
        [http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf](http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf).
    """

    def __init__(
        self,
        integrand,
        abs_tol=0.05,
        rmse_tol=None,
        n_init=256,
        n_limit=1e10,
        alpha=0.01,
        levels_min=2,
        levels_max=10,
    ):
        r"""
        Args:
            integrand (AbstractIntegrand): The integrand.
            abs_tol (np.ndarray): Absolute error tolerance.
            rmse_tol (np.ndarray): Root mean squared error tolerance.
                If supplied, then absolute tolerance and alpha are ignored in favor of the rmse tolerance.
            n_init (int): Initial number of samples.
            n_limit (int): Maximum number of samples.
            alpha (np.ndarray): Uncertainty level in $(0,1)$.
            levels_min (int): Minimum level of refinement $\geq 2$.
            levels_max (int): Maximum level of refinement $\geq$ `levels_min`.
        """
        self.parameters = ["rmse_tol", "n_init", "n_limit", "replications"]
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else:  # use absolute tolerance
            self.rmse_tol = float(abs_tol) / norm.ppf(1 - alpha / 2)
        self.alpha = alpha
        assert 0 < self.alpha < 1
        self.n_init = n_init
        self.n_limit = n_limit
        self.levels_min = levels_min
        self.levels_max = levels_max
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(CubMLQMC, self).__init__(
            allowed_distribs=[AbstractLDDiscreteDistribution],
            allow_vectorized_integrals=False,
        )
        self.replications = self.discrete_distrib.replications
        assert self.replications >= 4, "require at least 4 replications"

    def _validate_resume(self, data):
        self._validate_resume_data(data, required_fields=self._RESUME_REQUIRED_FIELDS)
        if np.shape(data.mean_level_reps) != (data.levels, int(self.replications)):
            raise ParameterError(
                "resume data mean_level_reps shape %s is incompatible with levels=%d, replications=%d."
                % (np.shape(data.mean_level_reps), data.levels, int(self.replications))
            )
        try:
            self._validate_level_replay_cache(data)
        except ValueError as exc:
            raise ParameterError(str(exc)) from exc

    def _restore_resume_state(self, data):
        # eval_level is all-False at end of a converged run; the loop's first
        # update_data call is a no-op, then the algorithm re-evaluates variance
        # and bias against the new rmse_tol (tighter or looser) and samples
        # accordingly.  With a looser tolerance the variance/bias conditions are
        # already met and the loop exits immediately.
        pass

    def _can_replay_resume_exactly(self, data):
        checkpoint_tol = self._checkpoint_rmse_tol(data)
        if checkpoint_tol is None or not (self.rmse_tol < checkpoint_tol):
            return False
        return hasattr(data, "level_rep_sums") and hasattr(data, "level_n_increments")

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

    def _run_integrate_loop(
        self,
        data,
        update_data_fn,
        trace=None,
        record_snapshots=False,
        stop_condition=None,
        t_start=None,
        resume_provenance=None,
    ):
        snapshots = []
        while True:
            update_data_fn(data)
            data.rmse_estimate = np.sqrt(data.var_level[:data.levels].sum() + data.bias_estimate**2)
            data.rmse_tol = self.rmse_tol
            if t_start is not None:
                self._set_elapsed_time(
                    data, time() - t_start, resume_provenance=resume_provenance
                )
            if record_snapshots:
                snapshots.append(copy.deepcopy(data))
            if stop_condition is not None and stop_condition(data):
                break
            if trace is not None:
                trace.iteration(data, step_value=int(data.levels))
            if data.var_level.sum() > (self.rmse_tol**2 / 2.0):
                efficient_level = np.argmax(data.var_cost_ratio_level)
                data.eval_level[efficient_level] = True
            elif data.bias_estimate > (self.rmse_tol / np.sqrt(2.0)):
                if data.levels == self.levels_max + 1:
                    warnings.warn(
                        "Failed to achieve weak convergence. levels == levels_max.",
                        MaxLevelsWarning,
                    )
                self._add_level(data)
            else:
                break
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
                break
        return snapshots

    def integrate(self, resume=None) -> tuple:
        """Run (or continue) the MLQMC integration.

        Args:
            resume (Data, optional): Checkpoint returned by a previous
                ``integrate()`` call.  The new tolerance may be tighter *or*
                looser than the one used when the checkpoint was created.
                With a tighter tolerance the algorithm draws additional samples
                from where it left off.  With a looser tolerance the existing
                samples already satisfy the requirement and the method returns
                immediately with no new sampling.

        Returns:
            tuple: ``(solution, data)``.
        """
        t_start = time()
        resume_provenance = self._capture_resume_provenance(resume)
        trace = self._make_trace_logger()
        data = self._prepare_resume_data(resume, self._validate_resume, self._restore_resume_state)
        if data is not None:
            data.rmse_estimate = np.sqrt(data.var_level[:data.levels].sum() + data.bias_estimate**2)
            self._set_elapsed_time(data, 0.0, resume_provenance=resume_provenance)
            trace.resume(data, step_value=int(data.levels))
        if data is None:
            data = self._construct_data()
        self._run_integrate_loop(
            data,
            self.update_data,
            trace=trace,
            t_start=t_start,
            resume_provenance=resume_provenance,
        )
        self._finalize_integration_data(
            data, time() - t_start, resume_provenance=resume_provenance
        )
        trace.finalize()
        data.iteration_history = getattr(self, "iteration_history", None)
        data.history_df = getattr(self, "history_df", None)
        return data.solution, data
